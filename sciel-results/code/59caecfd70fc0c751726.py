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
    
    # Precompute: for each vertex, how many of its children are at each depth
    # This helps estimate reload cost: a vertex used many times in upcoming levels
    # is expensive to evict
    children_by_depth = [defaultdict(int) for _ in range(n)]
    for v in range(n):
        for c in children[v]:
            children_by_depth[v][depth[c]] += 1
    
    # Compute "future_load_cost" for each vertex at each processing step:
    # If we evict v now, we'll need to reload it for each future child that
    # doesn't share a parent already in cache. Approximation: count remaining
    # children weighted by whether they share parents.
    # Simpler: reload cost ~ remaining_uses[v] (each future use may require a load)
    # But if children are consecutive, we load once. Use a heuristic.
    
    red = set()  # vertices currently in fast memory
    operations = []
    
    def eviction_cost(v):
        """
        Estimate the cost of keeping v in fast memory vs evicting.
        Lower cost = better to evict.
        
        Factors:
        - remaining_uses[v]: number of future children that need v
        - depth[v]: deeper vertices are closer to outputs, harder to recompute
        - Fan-out: vertices with many children at similar depths benefit from caching
        """
        if remaining_uses[v] == 0:
            return -1  # evict immediately
        
        # Cost of keeping: 1 (occupies a slot)
        # Cost of evicting: expected number of reloads
        # A vertex with R remaining uses will be loaded at most R times,
        # but if its children are processed in bursts, fewer loads.
        # Heuristic: reload_cost = remaining_uses[v] / max(1, burst_size)
        # where burst_size is the number of consecutive children at the same depth
        
        # Simple heuristic: higher remaining uses = more expensive to evict
        # Deeper vertices are more expensive to recompute if evicted
        # but we're not recomputing, just reloading from slow memory
        
        # Score: we want to evict vertices with LOW remaining uses and LOW depth
        # (low depth means they were inputs or early computations, cheaper to reload
        #  in the sense that they're simple)
        # Actually, all reloads cost 1 transfer. The key is:
        # - If remaining_uses is high, evicting causes many future loads
        # - If remaining_uses is low, evicting saves a slot for more valuable vertices
        
        # Use: remaining_uses * depth_weight
        # Vertices with high remaining uses should be kept
        # Break ties by preferring to keep deeper vertices (closer to outputs)
        return remaining_uses[v] * 10 - depth[v]
    
    def evict_lowest_priority():
        """Evict vertex with lowest priority (highest eviction benefit)"""
        if not red:
            return None
        best_vertex = None
        best_cost = float('inf')
        for v in red:
            c = eviction_cost(v)
            if c < best_cost:
                best_cost = c
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
    # Within each level, sort vertices to maximize cache locality:
    # Process vertices whose parents are already in red first
    
    for d in range(max_depth + 1):
        if d not in depth_groups:
            continue
        
        level_vertices = depth_groups[d]
        
        # Sort vertices: prefer those with more parents already in red
        # This reduces the number of loads needed
        def sort_key(v):
            if not parents_list[v]:
                return (0, 0)  # inputs first (no parents to load)
            in_cache = sum(1 for p in parents_list[v] if p in red)
            total_parents = len(parents_list[v])
            # Higher in_cache ratio = better
            return (-in_cache, total_parents - in_cache)
        
        level_vertices_sorted = sorted(level_vertices, key=sort_key)
        
        # Process vertices at this depth
        for v in level_vertices_sorted:
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
            
            # Proactive eviction: if any parent is no longer needed, evict it
            # to free up space for upcoming computations
            for p in parents_list[v]:
                if remaining_uses[p] == 0 and p in red:
                    operations.append(['store', p])
                    red.discard(p)
        
        # After processing each level, clean up vertices no longer needed
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
