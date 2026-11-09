"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import heapq
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
    
    # Precompute: for each depth level, which vertices are needed in future levels
    # needed_in_future[d] = set of vertices at depth d that have children at depth > d
    # Actually, we need to know which vertices will be needed again
    # A vertex v is "needed in future" if it has children that haven't been processed yet
    # Since we process in topological order, a vertex is needed in future if remaining_uses > 0
    
    # Sort vertices within each depth group by remaining uses (descending) to process
    # high-use vertices first, keeping them in cache longer
    for d in depth_groups:
        depth_groups[d].sort(key=lambda v: -remaining_uses[v])
    
    red = set()  # vertices currently in fast memory
    operations = []
    
    # Heap for eviction: (priority_key, version, vertex)
    # priority_key = (remaining_uses, -depth) - lower is evicted first
    # version tracks invalidation when remaining_uses changes
    eviction_heap = []
    version = [0] * n  # version counter for each vertex in heap
    
    def heap_push(v):
        """Push vertex v into eviction heap with current priority"""
        key = (remaining_uses[v], -depth[v])
        heapq.heappush(eviction_heap, (key[0], key[1], version[v], v))
        version[v] += 1
    
    def heap_pop_valid():
        """Pop from heap until we find a valid entry"""
        while eviction_heap:
            k0, k1, ver, v = heapq.heappop(eviction_heap)
            if ver == version[v] and v in red:
                return v
        return None
    
    def evict_lowest_priority():
        """Evict vertex with lowest priority from heap"""
        v = heap_pop_valid()
        if v is not None:
            operations.append(['store', v])
            red.discard(v)
    
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
                heap_push(v)
    
    def update_heap_entry(v):
        """Update heap entry for vertex v (called when remaining_uses changes)"""
        if v in red:
            heap_push(v)
    
    # Hierarchical pebbling: process by depth levels
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
                heap_push(v)
            
            # Decrement remaining uses for parents and update heap
            for p in parents_list[v]:
                remaining_uses[p] -= 1
                update_heap_entry(p)
        
        # After processing each level, evict vertices no longer needed
        # First, evict vertices with 0 remaining uses
        while len(red) > 0:
            # Check if lowest priority vertex has 0 uses
            if eviction_heap:
                k0, k1, ver, v = eviction_heap[0]
                if k0 == 0 and ver == version[v] and v in red:
                    evict_lowest_priority()
                    continue
            break
        
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
