import time
import sys
import numpy as np
from collections import deque

def pebble(dag, fast_memory_budget):
    start_time = time.time()
    deadline = start_time + 85.0  # 5s safety margin
    
    # 1. Parse DAG
    # dag is expected to have:
    # - 'vertices': list of vertex ids or range
    # - 'edges': list of (parent, child) or similar
    # - 'inputs': list of input vertex ids
    # - 'outputs': list of output vertex ids
    # Based on typical interfaces for such tasks, we might need to infer structure.
    # Assuming standard format:
    # dag['vertices'] : list of vertex indices
    # dag['edges'] : list of (u, v) where u is parent of v
    # dag['inputs'] : list of input indices
    # dag['outputs'] : list of output indices
    
    if 'vertices' not in dag or 'edges' not in dag:
        # Fallback or error
        return []

    vertices = dag['vertices']
    V = len(vertices)
    edges = dag['edges']
    inputs = set(dag.get('inputs', []))
    outputs = set(dag.get('outputs', []))
    
    # Map vertex ids to indices if necessary
    # Assuming vertices are 0..V-1 for simplicity, or we map them.
    # Let's assume vertices are integers 0..V-1.
    
    parents = [0] * V  # Bitset of parents
    children = [0] * V  # Bitset of children
    in_degree = [0] * V
    out_degree = [0] * V
    
    for u, v in edges:
        parents[v] |= (1 << u)
        children[u] |= (1 << v)
        in_degree[v] += 1
        out_degree[u] += 1
        
    # Calculate 'required' vertices: ancestors of outputs
    required_mask = 0
    # Topological sort to find ancestors
    # Since it's a DAG, we can do a reverse BFS from outputs.
    visited = [False] * V
    queue = deque(outputs)
    for o in outputs:
        visited[o] = True
        required_mask |= (1 << o)
        
    while queue:
        u = queue.popleft()
        # Find parents of u
        p_mask = parents[u]
        # Iterate through set bits in p_mask
        p = p_mask
        while p:
            lsb = p & -p
            idx = lsb.bit_length() - 1
            if not visited[idx]:
                visited[idx] = True
                required_mask |= (1 << idx)
                queue.append(idx)
            p ^= lsb
            
    # Priority for vertices: 
    # Higher priority if it has more children (reduces loads)
    # or is an output.
    priority = [0] * V
    for v in range(V):
        if v in outputs:
            priority[v] = 1000 + out_degree[v]
        else:
            priority[v] = out_degree[v]
            
    # State
    in_fast = 0
    on_disk = 0
    for i in inputs:
        on_disk |= (1 << i)
        
    ops = []
    
    # Helper to check if goal reached
    def is_goal():
        return (required_mask & outputs) == (on_disk & outputs) # All outputs are on disk
        
    # Actually, goal is all outputs in on_disk.
    # (outputs_mask & ~on_disk) == 0
    
    outputs_mask = 0
    for o in outputs:
        outputs_mask |= (1 << o)
        
    def check_goal():
        return (outputs_mask & ~on_disk) == 0
        
    # Greedy Loop
    # We track the number of transfers
    transfers = 0
    
    # To avoid infinite loops, track visited states or just rely on termination.
    # Since we always make progress (either compute a new vertex or store/delete), it should terminate.
    
    while not check_goal():
        if time.time() > deadline:
            break
            
        # 1. Identify ready vertices: parents are available (in_fast or on_disk)
        # and the vertex is not already in in_fast or on_disk (if on_disk, we might load it, but if it's computed and on disk, we don't need to recompute).
        # Actually, if v is in on_disk, it's "available". We can load it.
        # If v is not in in_fast and not in on_disk, we must compute it.
        
        ready_to_compute = []
        ready_to_load = []
        
        for v in range(V):
            if v in in_fast or (v in on_disk and not (required_mask & (1<<v))):
                continue # Already in fast, or not required and on disk? 
                # If not required, we don't care.
            
            if not (required_mask & (1 << v)):
                continue # Not an ancestor of output, ignore
                
            # Check if parents are available
            p_mask = parents[v]
            available = in_fast | on_disk
            if (p_mask & ~available) == 0:
                # All parents are available
                if (in_fast & (1 << v)) == 0:
                    if (on_disk & (1 << v)) != 0:
                        ready_to_load.append(v)
                    else:
                        ready_to_compute.append(v)
                        
        # Sort by priority
        ready_to_compute.sort(key=lambda x: priority[x], reverse=True)
        ready_to_load.sort(key=lambda x: priority[x], reverse=True)
        
        action_taken = False
        
        # Try to compute or load the highest priority ready vertex
        # We need space in in_fast.
        
        # If in_fast is full, we must evict.
        # Eviction strategy:
        # Find the vertex in in_fast with the lowest priority.
        # If it has unprocessed children, store it. Else delete it.
        
        def evict_lowest():
            nonlocal in_fast, on_disk, transfers, ops
            min_prio = float('inf')
            victim = -1
            for v in range(V):
                if (in_fast & (1 << v)):
                    if priority[v] < min_prio:
                        min_prio = priority[v]
                        victim = v
            if victim != -1:
                # Check if victim has unprocessed children
                # Unprocessed children: children not in on_disk and not in in_fast?
                # Or simply children not yet "satisfied".
                # If we store it, it becomes available on disk.
                # If we delete it, it's gone.
                # If it has children that are not yet computed, we should store it to avoid recompute.
                c_mask = children[victim]
                # Count children that are not yet in on_disk
                # Actually, if a child is not in on_disk, it might need this parent.
                # Let's assume we store if out_degree[victim] > 0.
                if out_degree[victim] > 0:
                    # Store
                    on_disk |= (1 << victim)
                    in_fast &= ~(1 << victim)
                    ops.append(['store', victim])
                    transfers += 1
                else:
                    # Delete
                    in_fast &= ~(1 << victim)
                    ops.append(['delete', victim])
                    # transfers += 0
                    
        # If we have ready vertices, try to process one
        if ready_to_compute or ready_to_load:
            # Pick the best candidate
            # If in_fast is full, evict first
            if bin(in_fast).count('1') >= fast_memory_budget:
                evict_lowest()
                
            # Now pick the best ready vertex
            # Prefer compute over load? Compute is free. Load costs 1.
            # But if we have to load, we pay 1.
            # Let's pick the highest priority from both lists, but penalize load?
            # Or just pick the highest priority overall.
            
            best_v = -1
            is_load = False
            
            if ready_to_compute:
                best_v = ready_to_compute[0]
                is_load = False
            if ready_to_load:
                # Compare priority
                if not ready_to_compute or priority[ready_to_load[0]] > priority[ready_to_compute[0]]:
                     # If load has higher priority, maybe we should load?
                     # But compute is free.
                     # Let's stick to compute if available, as it saves a transfer.
                     pass
            
            # Actually, if we can compute, we should.
            # If we can only load, we load.
            
            if ready_to_compute:
                v = ready_to_compute[0]
                # Compute
                in_fast |= (1 << v)
                ops.append(['compute', v])
                # transfers += 0
                action_taken = True
            elif ready_to_load:
                v = ready_to_load[0]
                # Load
                in_fast |= (1 << v)
                on_disk &= ~(1 << v) # Remove from disk? Or keep? 
                # Usually, load moves from disk to memory. Disk copy remains?
                # "load/store" implies movement.
                # If we load, is it removed from disk?
                # In pebbling, usually blue pebble (disk) and red pebble (memory) are distinct.
                # If we load, we put a red pebble. The blue pebble stays?
                # If the blue pebble stays, we can store back without recompute.
                # Let's assume load does NOT remove from disk.
                # ops.append(['load', v])
                ops.append(['load', v])
                transfers += 1
                action_taken = True
                
        if not action_taken:
            # No ready vertices?
            # This can happen if we have a cycle (not a DAG) or if we are stuck.
            # Or if we need to load a parent that is on disk but we didn't identify it as ready?
            # Wait, if parent is on disk, it is available.
            # So if no vertex is ready, it means no parent is available?
            # This implies we are stuck.
            # Break to avoid infinite loop.
            break
            
    return ops

# EVOLVE-BLOCK-END
