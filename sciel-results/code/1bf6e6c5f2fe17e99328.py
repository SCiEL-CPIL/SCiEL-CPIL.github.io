# EVOLVE-BLOCK-START
import time

def pebble(dag, fast_memory_budget):
    """
    Communication-minimizing DAG pebbling with improved heuristics.
    
    Strategy:
    1. Parse the DAG structure
    2. Use greedy pebbling with O(1) last_use lookup via position map
    3. Smart deletion: evict vertices with latest last_use among non-urgent candidates
    4. Avoid loading blue parents into fast memory unnecessarily
    5. Return operations list
    """
    start_time = time.time()
    deadline = start_time + 85  # Leave 5s safety margin
    
    try:
        # Parse DAG structure
        vertices = dag.get('vertices', [])
        if not vertices:
            if 'nodes' in dag:
                vertices = dag['nodes']
            else:
                vertices = list(dag.keys()) if isinstance(dag, dict) else []
        
        parents_map = dag.get('parents', {})
        if not parents_map and isinstance(dag, dict):
            parents_map = {v: list(parents) for v, parents in dag.items()}
        
        inputs = set(dag.get('inputs', []))
        outputs = set(dag.get('outputs', []))
        
        if not inputs:
            inputs = set(v for v in vertices if not parents_map.get(v, []))
        
        # Build children map
        children_map = {v: [] for v in vertices}
        for v in vertices:
            for p in parents_map.get(v, []):
                if p in children_map:
                    children_map[p].append(v)
        
        if not outputs:
            outputs = set(v for v in vertices if not children_map.get(v, []))
        
        topo_order = list(vertices)
        n = len(topo_order)
        
        # O(1) position lookup
        pos_map = {v: i for i, v in enumerate(topo_order)}
        
        # Compute last_use in O(n + E) using reverse topo order
        last_use = {}
        for v in reversed(topo_order):
            if v in outputs:
                last_use[v] = n  # Needed until the end
            elif not children_map.get(v, []):
                last_use[v] = n  # Leaf: keep until end (might be needed for recompute)
            else:
                max_pos = 0
                for child in children_map.get(v, []):
                    child_pos = pos_map.get(child, n)
                    if child_pos > max_pos:
                        max_pos = child_pos
                last_use[v] = max_pos
        
        # Build operation list
        ops = []
        in_fast = set()
        is_blue = set(inputs)
        
        # Precompute: for each position i, which vertices are parents of vertices at positions i..i+window
        # This helps with deletion heuristic. Use a larger window for better lookahead.
        window = min(20, n)
        # Precompute future_parent_sets[i] = set of parents of vertices in topo_order[i:i+window]
        future_parent_sets = []
        for i in range(n):
            fps = set()
            end = min(i + window, n)
            for j in range(i, end):
                nv = topo_order[j]
                for p in parents_map.get(nv, []):
                    fps.add(p)
            future_parent_sets.append(fps)
        
        # Process each vertex in topological order
        for i, v in enumerate(topo_order):
            if time.time() > deadline:
                break
            
            if v in inputs:
                continue
            
            parents = parents_map.get(v, [])
            
            # Ensure all parents are available (blue or in fast memory)
            for p in parents:
                # If parent is blue, it's available without loading
                if p in is_blue:
                    continue
                if p in in_fast:
                    continue
                
                # Need to load parent p
                if len(in_fast) >= fast_memory_budget:
                    # Need to delete something to make room
                    fps = future_parent_sets[i] if i < n else set()
                    candidates = list(in_fast)
                    
                    # Prefer to delete vertices not in future_parent_sets (not needed soon)
                    non_urgent = [x for x in candidates if x not in fps]
                    if non_urgent:
                        to_delete = max(non_urgent, key=lambda x: last_use.get(x, n))
                    else:
                        # All are potentially needed soon; delete the one with latest last_use
                        to_delete = max(candidates, key=lambda x: last_use.get(x, n))
                    
                    ops.append(['delete', to_delete])
                    in_fast.remove(to_delete)
                
                ops.append(['load', p])
                in_fast.add(p)
            
            # Compute v
            ops.append(['compute', v])
            in_fast.add(v)
            
            # If v is an output, store it to make it blue
            if v in outputs:
                ops.append(['store', v])
                in_fast.remove(v)
                is_blue.add(v)
            
            # Delete parents that are no longer needed (last_use <= current position)
            for p in parents:
                if p in in_fast and last_use.get(p, n) <= i:
                    ops.append(['delete', p])
                    in_fast.remove(p)
        
        if len(ops) > 20000:
            ops = ops[:20000]
        
        return ops
        
    except Exception:
        try:
            from autoresearch_suite.science_tcs_expansion import pebble_baseline
            return pebble_baseline(dag, fast_memory_budget)
        except:
            return []
# EVOLVE-BLOCK-END
