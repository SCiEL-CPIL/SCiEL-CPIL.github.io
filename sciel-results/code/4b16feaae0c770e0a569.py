"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import heapq
from collections import deque

def pebble(dag, fast_memory_budget):
    # Robustly extract graph structure
    # Assume dag has .nodes, .parents, .outputs or similar
    # Fallback to common patterns
    
    nodes = []
    parents = {}
    outputs = []
    
    # Try common attribute names
    if hasattr(dag, 'nodes'):
        nodes = list(dag.nodes)
    elif isinstance(dag, list):
        nodes = dag
    elif isinstance(dag, dict):
        nodes = list(dag.keys())
        
    if hasattr(dag, 'parents'):
        parents = {k: list(v) for k, v in dag.parents.items()}
    elif hasattr(dag, 'edges'):
        # If edges are given, construct parents
        for u, v in dag.edges:
            if v not in parents:
                parents[v] = []
            parents[v].append(u)
            
    if hasattr(dag, 'outputs'):
        outputs = list(dag.outputs)
    elif hasattr(dag, 'sinks'):
        outputs = list(dag.sinks)
        
    # If no outputs defined, assume all nodes with no children are outputs?
    # Or maybe the last layer?
    # Let's assume if outputs is empty, we treat all nodes as potential outputs 
    # but only those that are sinks need to be blue? 
    # "All outputs must finish blue". 
    # If no outputs specified, maybe the task implies specific nodes.
    # For safety, if outputs is empty, we will not force store on non-outputs.
    
    # Build children map
    children = {n: [] for n in nodes}
    for v, ps in parents.items():
        for p in ps:
            children[p].append(v)
            
    # Topological Sort (Kahn's algorithm)
    in_degree = {n: len(parents.get(n, [])) for n in nodes}
    queue = deque([n for n in nodes if in_degree[n] == 0])
    topo_order = []
    
    while queue:
        n = queue.popleft()
        topo_order.append(n)
        for c in children[n]:
            in_degree[c] -= 1
            if in_degree[c] == 0:
                queue.append(c)
                
    # If topo_order length != nodes length, there's a cycle or disconnected components.
    # Assume valid DAG.
    
    # Precompute last_child_index for each node
    # last_child_index[n] = index in topo_order of the last child of n
    # If n has no children, last_child_index[n] = index of n itself
    node_to_index = {n: i for i, n in enumerate(topo_order)}
    last_child_index = {}
    for n in nodes:
        if children[n]:
            # Find the child with the maximum topo index
            max_idx = max(node_to_index[c] for c in children[n])
            last_child_index[n] = max_idx
        else:
            last_child_index[n] = node_to_index[n]
            
    # Remaining children count for cleanup
    remaining_children = {n: len(children[n]) for n in nodes}
    
    # State
    in_fast = set()
    ops = []
    
    # Eviction helper
    def evict_one(exclude=set()):
        """Evict one node from in_fast to make space.
        Returns the evicted node id.
        """
        if not in_fast:
            return None
        
        # Find candidate to evict
        # Heuristic: Evict node with largest last_child_index (needed furthest in future)
        # Exclude nodes that are parents of the current node (we need them now)
        candidates = [n for n in in_fast if n not in exclude]
        
        if not candidates:
            # Fallback: evict any node (should not happen if we check size correctly)
            candidates = list(in_fast)
            
        # Pick max last_child_index
        victim = max(candidates, key=lambda x: last_child_index[x])
        
        # Perform eviction
        # If victim is an output, we must store it
        if victim in outputs:
            ops.append(['store', victim])
        
        # Delete frees the slot
        ops.append(['delete', victim])
        in_fast.remove(victim)
        
        return victim

    # Main Loop
    for v in topo_order:
        # 1. Ensure parents are in fast memory
        ps = parents.get(v, [])
        for p in ps:
            if p not in in_fast:
                # Need to load p
                # Check if fast memory is full
                while len(in_fast) >= fast_memory_budget:
                    # Evict a node
                    # We cannot evict p (we are loading it) or other parents of v (we need them)
                    # Actually, we are iterating parents one by one.
                    # We should ensure ALL parents are loaded before computing v.
                    # So when loading p, we might need to evict other nodes.
                    # But we must not evict other parents of v that are already in in_fast?
                    # If we evict a parent q of v, we'll have to reload it.
                    # It's better to evict non-parents.
                    exclude_set = set(ps) # All parents of v
                    evict_one(exclude_set)
                
                ops.append(['load', p])
                in_fast.add(p)
        
        # 2. Compute v
        ops.append(['compute', v])
        in_fast.add(v)
        
        # 3. Cleanup parents
        for p in ps:
            remaining_children[p] -= 1
            if remaining_children[p] == 0:
                # p is no longer needed
                if p in in_fast:
                    if p in outputs:
                        ops.append(['store', p])
                    ops.append(['delete', p])
                    in_fast.remove(p)
                    
    # 4. Final Cleanup: Ensure all outputs are in slow memory (Blue)
    # Any output still in in_fast must be stored.
    for v in outputs:
        if v in in_fast:
            ops.append(['store', v])
            ops.append(['delete', v])
            in_fast.remove(v)
            
    return ops
# EVOLVE-BLOCK-END
