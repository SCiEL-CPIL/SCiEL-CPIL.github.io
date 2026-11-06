"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from collections import defaultdict

def pebble(dag, fast_memory_budget):
    """
    Communication-minimizing DAG pebbling.
    dag structure expected:
      - dag['parents']: dict vertex -> list of parent vertices (topologically ordered)
      - dag['outputs']: list of output vertices
      - dag['inputs']: list of input vertices (start blue)
    Returns list of [operation, vertex] where operation in {load, store, compute, delete}.
    """
    start_time = time.time()
    DEADLINE = 85.0  # wall-clock limit with safety margin
    
    try:
        parents = dag.get('parents', {})
        outputs = dag.get('outputs', [])
        inputs = dag.get('inputs', [])
        
        # Build reverse adjacency (children)
        children = defaultdict(list)
        for v, ps in parents.items():
            for p in ps:
                children[p].append(v)
        
        all_vertices = set(parents.keys()) | set(inputs) | set(outputs)
        
        # Topological order
        in_degree = {v: len(parents.get(v, [])) for v in all_vertices}
        for v in inputs:
            in_degree[v] = 0
        
        queue = [v for v in all_vertices if in_degree.get(v, 0) == 0]
        topo_order = []
        while queue:
            queue.sort()
            u = queue.pop(0)
            topo_order.append(u)
            for c in children.get(u, []):
                in_degree[c] -= 1
                if in_degree[c] == 0:
                    queue.append(c)
        
        # Position in topological order for quick lookup
        topo_pos = {v: i for i, v in enumerate(topo_order)}
        
        # Count how many times each vertex is needed as a parent
        # remaining_uses[v] = number of children of v that haven't been processed yet
        remaining_uses = defaultdict(int)
        for v in all_vertices:
            ps = parents.get(v, [])
            for p in ps:
                remaining_uses[p] += 1
        
        # Outputs must be in fast memory at the end
        must_be_blue = set(outputs)
        
        ops = []
        fast_mem = set()  # vertices currently in fast memory (red)
        processed = set()  # vertices that have been computed/loaded
        
        def can_add_to_fast_mem(v):
            """Check if we can add vertex v to fast memory without exceeding budget."""
            if len(fast_mem) < fast_memory_budget:
                return True
            # If we need to evict, check if there's a vertex we can evict
            # We'll handle eviction separately
            return False
        
        def find_evictable_vertex():
            """Find a vertex in fast memory that can be safely evicted (no remaining uses and not an output)."""
            candidates = []
            for v in fast_mem:
                if v in must_be_blue:
                    continue
                if remaining_uses[v] == 0:
                    candidates.append(v)
            if candidates:
                # Evict the one with least remaining uses (all zero here)
                return candidates[0]
            # If no vertex with zero remaining uses, evict one with fewest remaining uses
            # This is a heuristic - might cause recomputation
            non_output = [v for v in fast_mem if v not in must_be_blue]
            if non_output:
                # Sort by remaining_uses ascending, then by topo_pos descending (least important first)
                non_output.sort(key=lambda v: (remaining_uses[v], -topo_pos.get(v, 0)))
                return non_output[0]
            return None
        
        # Process in topological order
        for v in topo_order:
            if time.time() > DEADLINE:
                break
            
            ps = parents.get(v, [])
            
            if not ps:
                # Input vertex - start blue, need to load into fast memory
                if v not in fast_mem:
                    # Check if we need to evict
                    if len(fast_mem) >= fast_memory_budget:
                        evict = find_evictable_vertex()
                        if evict is not None:
                            ops.append(['store', evict])
                            fast_mem.discard(evict)
                    if v not in fast_mem:
                        ops.append(['load', v])
                        fast_mem.add(v)
                processed.add(v)
            else:
                # Compute vertex - need all parents in fast memory
                for p in ps:
                    if p not in fast_mem:
                        # Need to load parent from slow memory
                        if len(fast_mem) >= fast_memory_budget:
                            evict = find_evictable_vertex()
                            if evict is not None:
                                ops.append(['store', evict])
                                fast_mem.discard(evict)
                        if p not in fast_mem:
                            ops.append(['load', p])
                            fast_mem.add(p)
                
                # Compute v
                ops.append(['compute', v])
                fast_mem.add(v)
                processed.add(v)
                
                # After computing v, decrement remaining_uses for its parents
                for p in ps:
                    remaining_uses[p] -= 1
                    # Check if we should delete p (no more uses and not an output)
                    if remaining_uses[p] == 0 and p not in must_be_blue and p in fast_mem:
                        ops.append(['delete', p])
                        fast_mem.discard(p)
        
        # Ensure all outputs are in fast memory at the end
        for o in outputs:
            if o not in fast_mem:
                if len(fast_mem) >= fast_memory_budget:
                    evict = find_evictable_vertex()
                    if evict is not None:
                        ops.append(['store', evict])
                        fast_mem.discard(evict)
                if o not in fast_mem:
                    ops.append(['load', o])
                    fast_mem.add(o)
        
        # Check operation count
        if len(ops) <= 20000:
            return ops
        else:
            # Too many ops, truncate
            return ops[:20000]
            
    except Exception as e:
        return []
# EVOLVE-BLOCK-END
