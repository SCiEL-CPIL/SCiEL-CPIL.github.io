"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from heapq import heappush, heappop
from collections import defaultdict

def pebble(dag, fast_memory_budget):
    start_time = time.time()
    deadline = 85.0  # seconds, safety margin before 90s kill
    
    # Parse the DAG structure
    if hasattr(dag, 'num_vertices'):
        n = dag.num_vertices
    elif hasattr(dag, 'n'):
        n = dag.n
    elif isinstance(dag, dict) and 'n' in dag:
        n = dag['n']
    else:
        if hasattr(dag, 'vertices'):
            n = len(dag.vertices)
        else:
            from autoresearch_suite.science_tcs_expansion import pebble_baseline
            return pebble_baseline(dag, fast_memory_budget)
    
    def get_parents(v):
        if hasattr(dag, 'parents'):
            return dag.parents[v]
        elif isinstance(dag, dict) and 'parents' in dag:
            return dag['parents'][v]
        else:
            return []
    
    def get_outputs():
        if hasattr(dag, 'outputs'):
            return set(dag.outputs)
        elif isinstance(dag, dict) and 'outputs' in dag:
            return set(dag['outputs'])
        else:
            return {n-1}
    
    outputs = get_outputs()
    
    # Build efficient data structures
    parents_list = [get_parents(v) for v in range(n)]
    
    # Compute children for each vertex
    children = [[] for _ in range(n)]
    for v in range(n):
        for p in parents_list[v]:
            children[p].append(v)
    
    # Compute remaining uses (how many children still need this vertex)
    remaining_uses = [0] * n
    for v in range(n):
        remaining_uses[v] = len(children[v])
    
    # Compute depth (topological level) for hierarchical decomposition
    depth = [0] * n
    for v in range(n):
        if parents_list[v]:
            depth[v] = max(depth[p] for p in parents_list[v]) + 1
    
    # Group vertices by depth for hierarchical processing
    depth_groups = defaultdict(list)
    for v in range(n):
        depth_groups[depth[v]].append(v)
    
    max_depth = max(depth) if depth > 0 else 0
    
    red = set()  # vertices currently in fast memory
    operations = []
    
    def evict_lowest_priority():
        """Evict vertex with lowest remaining uses, break ties by depth"""
        if not red:
            return None
        best_vertex = None
        best_key = (float('inf'), float('inf'))
        for v in red:
            # Priority: (remaining_uses, -depth) - prefer evicting low-use, low-depth
            key = (remaining_uses[v], -depth[v])
            if key < best_key:
                best_key = key
                best_vertex = v
        if best_vertex is not None:
            operations.append(['store', best_vertex])
            red.discard(best_vertex)
    
    def ensure_space():
        """Ensure fast memory has space for one more vertex"""
        while len(red) >= fast_memory_budget:
            evict_lowest_priority()
    
    def load_vertex(v):
        """Load vertex v into fast memory if not already there"""
        if v not in red:
            ensure_space()
            if v not in red:
                operations.append(['load', v])
                red.add(v)
    
    # Hierarchical pebbling: process by depth levels
    # At each level, we can keep vertices in fast memory for use in next level
    # Strategy: keep vertices with high remaining uses in fast memory
    
    for d in range(max_depth + 1):
        if d not in depth_groups:
            continue
        
        level_vertices = depth_groups[d]
        
        # Process vertices at this depth
        for v in level_vertices:
            if time.time() - start_time > deadline - 10:
                break
            
            if not parents_list[v]:
                # Input vertex: load from slow to fast
                load_vertex(v)
            else:
                # Has parents: need all parents in fast memory to compute
                for p in parents_list[v]:
                    if p not in red:
                        load_vertex(p)
                
                # Compute v
                operations.append(['compute', v])
                red.add(v)
            
            # Decrement remaining uses for parents
            for p in parents_list[v]:
                remaining_uses[p] -= 1
        
        # After processing each level, consider evicting vertices no longer needed
        # Keep only vertices that will be used in future levels
        if len(red) > fast_memory_budget:
            # Evict vertices with 0 remaining uses first
            zero_use = [v for v in red if remaining_uses[v] == 0]
            for v in zero_use:
                operations.append(['store', v])
                red.discard(v)
            
            # If still over budget, evict lowest priority
            while len(red) >= fast_memory_budget:
                evict_lowest_priority()
    
    # Ensure all outputs are blue (stored back to slow memory)
    for v in outputs:
        if v in red:
            operations.append(['store', v])
            red.discard(v)
    
    # Check time and operation limit
    if len(operations) > 20000:
        operations = operations[:20000]
        for v in outputs:
            if v in red:
                operations.append(['store', v])
    
    return operations
# EVOLVE-BLOCK-END
