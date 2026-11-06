"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from collections import defaultdict, deque

def pebble(dag, fast_memory_budget):
    """
    Minimize communication (load+store ops) for DAG pebbling.
    
    dag: dict with keys:
        'vertices': list of vertex ids (topologically ordered)
        'parents': dict mapping vertex -> list of parent vertex ids
        'outputs': list of output vertex ids
    Inputs (vertices with no parents) start blue (in fast memory).
    All outputs must finish blue.
    
    Returns: list of [operation, vertex] where operation in {load, store, compute, delete}
    """
    deadline = time.time() + 50  # safety margin before 60s kill
    
    # Parse dag structure
    vertices = list(dag.get('vertices', []))
    if not vertices and isinstance(dag, dict):
        # Try to infer from parents
        parents = dag.get('parents', {})
        if parents:
            vertices = list(parents.keys())
    parents_map = dict(dag.get('parents', {})) if isinstance(dag.get('parents', {}), dict) else {}
    outputs = set(dag.get('outputs', []))
    
    # Build children map
    children_map = defaultdict(list)
    for v, ps in parents_map.items():
        for p in ps:
            children_map[p].append(v)
    
    # Count how many times each vertex is needed (fan-out + is_output)
    # A vertex is needed if it has children or is an output
    remaining_uses = {}
    for v in vertices:
        uses = len(children_map[v])
        if v in outputs:
            uses += 1  # must keep until end
        remaining_uses[v] = uses
    
    # Track fast memory state
    in_memory = set()
    operations = []
    op_count = 0
    MAX_OPS = 20000
    
    def op_budget_left():
        return op_count < MAX_OPS
    
    # Input vertices (no parents) start blue
    input_vertices = [v for v in vertices if not parents_map.get(v, [])]
    for v in input_vertices:
        if len(in_memory) < fast_memory_budget:
            in_memory.add(v)
        # If budget is 0, we still need to handle this
        if len(in_memory) >= fast_memory_budget and fast_memory_budget > 0:
            # Shouldn't happen for inputs if budget >= num inputs, but handle gracefully
            pass
    
    # If budget is 0, we can't keep anything - special case
    if fast_memory_budget == 0:
        # Every compute requires loading all parents, and every result must be stored immediately
        # except outputs which... can't be kept. This is degenerate.
        for v in vertices:
            if time.time() > deadline:
                break
            ps = parents_map.get(v, [])
            for p in ps:
                if p not in in_memory:
                    operations.append(['load', p])
                    op_count += 1
                    if not op_budget_left():
                        return operations
            operations.append(['compute', v])
            op_count += 1
            if not op_budget_left():
                return operations
            # Must store since no memory
            if v not in outputs or len(in_memory) >= 0:
                # store everything since no fast memory
                operations.append(['store', v])
                op_count += 1
                if not op_budget_left():
                    return operations
        # Ensure outputs are blue - but with budget 0, they're not in memory
        # This is a constraint violation if budget=0 and there are outputs
        # We'll try to load them at end if possible
        for v in outputs:
            if v not in in_memory:
                operations.append(['load', v])
                op_count += 1
                in_memory.add(v)
        return operations
    
    # Process vertices in topological order
    for v in vertices:
        if time.time() > deadline:
            break
        if not op_budget_left():
            break
            
        ps = parents_map.get(v, [])
        
        # Ensure all parents are in memory
        for p in ps:
            if p not in in_memory:
                operations.append(['load', p])
                op_count += 1
                in_memory.add(p)
                if not op_budget_left():
                    return operations
        
        # Compute v
        operations.append(['compute', v])
        op_count += 1
        if not op_budget_left():
            return operations
        
        # Add v to fast memory
        in_memory.add(v)
        
        # Update remaining uses for parents (they've been used by v)
        for p in ps:
            if p in remaining_uses:
                remaining_uses[p] -= 1
        
        # Check if we exceed budget - need to evict
        while len(in_memory) > fast_memory_budget:
            # Find vertex to evict: minimum remaining_uses, prefer non-outputs
            best_evict = None
            best_key = None
            for u in in_memory:
                # We don't want to evict v just computed if possible
                if u == v:
                    key = (1, remaining_uses.get(u, 0))  # deprioritize
                else:
                    key = (0, remaining_uses.get(u, 0))
                if best_key is None or key < best_key:
                    best_key = key
                    best_evict = u
            
            if best_evict is None:
                break
                
            # Evict best_evict
            operations.append(['store', best_evict])
            op_count += 1
            in_memory.discard(best_evict)
            if not op_budget_left():
                return operations
    
    # Ensure all outputs are in fast memory at the end
    for v in outputs:
        if v not in in_memory:
            operations.append(['load', v])
            op_count += 1
            in_memory.add(v)
            if not op_budget_left():
                break
    
    return operations
# EVOLVE-BLOCK-END
