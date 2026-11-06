"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import heapq
from collections import defaultdict, deque

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
    DEADLINE = 85.0  # wall-clock limit from evaluator with safety margin
    
    # Try to import baseline as fallback
    try:
        from autoresearch_suite.science_tcs_expansion import pebble_baseline
    except Exception:
        pebble_baseline = None

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
        
        # Map vertex to index in topo_order for O(1) lookup
        topo_index = {v: i for i, v in enumerate(topo_order)}
        
        # Precompute: for each vertex, the set of children (direct dependents)
        # and the "last_use" index: the highest topo_index of any child
        last_use = {}
        for v in all_vertices:
            if v in children and children[v]:
                last_use[v] = max(topo_index[c] for c in children[v])
            else:
                last_use[v] = topo_index.get(v, -1)
        
        # Also ensure outputs are kept until the end
        must_be_blue = set(outputs)
        
        # Precompute depth (recompute cost proxy)
        depth = {}
        for v in topo_order:
            ps = parents.get(v, [])
            if not ps:
                depth[v] = 0
            else:
                depth[v] = max(depth[p] for p in ps) + 1
        
        # Precompute future uses
        # remaining_uses[v] will be decremented as we process children
        remaining_uses = {v: len(children[v]) for v in all_vertices}
        
        ops = []
        fast_mem = set()
        
        def get_eviction_candidate():
            """
            Find the vertex in fast_mem to delete to make space.
            Criteria:
            1. Prefer deleting 'dead' vertices (remaining_uses == 0 and not output).
            2. Among live non-outputs, delete the one with lowest remaining_uses.
            3. Tie-breaker: lowest depth (easiest to recompute).
            4. Tie-breaker: earliest last_use.
            5. If only outputs remain, delete the one with lowest depth.
            """
            dead_candidates = []
            live_candidates = []
            output_candidates = []
            
            for v in fast_mem:
                if v in must_be_blue:
                    output_candidates.append((depth[v], last_use[v], v))
                else:
                    if remaining_uses[v] == 0:
                        dead_candidates.append((depth[v], last_use[v], v))
                    else:
                        live_candidates.append((remaining_uses[v], depth[v], last_use[v], v))
            
            # 1. Dead vertices are best to evict
            if dead_candidates:
                dead_candidates.sort(key=lambda x: (x[0], x[1]))
                return dead_candidates[0][2]
            
            # 2. Live non-outputs
            if live_candidates:
                # Sort by remaining_uses (ascending), then depth (ascending), then last_use (ascending)
                live_candidates.sort(key=lambda x: (x[0], x[1], x[2]))
                return live_candidates[0][3]
            
            # 3. Outputs
            if output_candidates:
                output_candidates.sort(key=lambda x: (x[0], x[1]))
                return output_candidates[0][2]
            
            return None

        def ensure_in_fast_mem(v):
            """
            Ensure vertex v is in fast_mem.
            Returns True if successful, False if budget exceeded and couldn't free space.
            """
            if v in fast_mem:
                return True
            
            ps = parents.get(v, [])
            if not ps:
                # It's an input
                if len(fast_mem) >= fast_memory_budget:
                    cand = get_eviction_candidate()
                    if cand is None:
                        return False
                    ops.append(['delete', cand])
                    fast_mem.discard(cand)
                
                ops.append(['load', v])
                fast_mem.add(v)
                return True
            else:
                # It's a computed vertex
                # First, ensure all its parents are in fast_mem
                for p in ps:
                    if not ensure_in_fast_mem(p):
                        return False
                
                # Now compute v
                if len(fast_mem) >= fast_memory_budget:
                    cand = get_eviction_candidate()
                    if cand is None:
                        return False
                    ops.append(['delete', cand])
                    fast_mem.discard(cand)
                
                ops.append(['compute', v])
                fast_mem.add(v)
                return True
        
        # Process in topological order
        for i, v in enumerate(topo_order):
            if time.time() > DEADLINE:
                break
            
            if not ensure_in_fast_mem(v):
                # Failed to get v in fast mem, break?
                break
            
            # Update remaining_uses for parents of v
            for p in parents.get(v, []):
                if p in remaining_uses:
                    remaining_uses[p] -= 1
            
            # Optional: Delete vertices that are no longer needed to save memory for future?
            # We can proactively delete vertices whose remaining_uses is 0 and are not outputs.
            # This keeps fast_mem small and reduces pressure.
            for u in list(fast_mem):
                if u in must_be_blue:
                    continue
                if remaining_uses[u] == 0:
                    ops.append(['delete', u])
                    fast_mem.discard(u)
        
        # Ensure all outputs are in fast_mem at the end
        for o in outputs:
            if o not in fast_mem:
                if not ensure_in_fast_mem(o):
                    # Try to evict and load
                    if len(fast_mem) >= fast_memory_budget:
                        cand = get_eviction_candidate()
                        if cand:
                            ops.append(['delete', cand])
                            fast_mem.discard(cand)
                    ensure_in_fast_mem(o)
        
        # Check operation count
        if len(ops) <= 20000:
            return ops
        else:
            # Too many ops, try baseline or truncate
            if pebble_baseline:
                return pebble_baseline(dag, fast_memory_budget)
            return ops[:20000]
            
    except Exception as e:
        # Fallback to baseline if available
        if pebble_baseline:
            return pebble_baseline(dag, fast_memory_budget)
        return []
# EVOLVE-BLOCK-END
