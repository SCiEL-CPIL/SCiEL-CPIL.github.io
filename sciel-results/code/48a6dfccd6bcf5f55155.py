"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from collections import defaultdict, deque

def pebble(dag, fast_memory_budget):
    start_time = time.time()
    DEADLINE = start_time + 50.0

    def elapsed():
        return time.time() - start_time

    def time_check():
        return elapsed() > DEADLINE

    # 1. Parse DAG
    try:
        if isinstance(dag, dict):
            parents = dict(dag.get('parents', {}))
            outputs = list(dag.get('outputs', []))
            vertices = dag.get('vertices', None)
            inputs = dag.get('inputs', None)
        else:
            parents = dict(getattr(dag, 'parents', {}))
            outputs = list(getattr(dag, 'outputs', []))
            vertices = getattr(dag, 'vertices', None)
            inputs = getattr(dag, 'inputs', None)
    except Exception:
        # Fallback to baseline if parsing fails
        try:
            from autoresearch_suite.science_tcs_expansion import pebble_baseline
            return pebble_baseline(dag, fast_memory_budget)
        except ImportError:
            raise

    if vertices is not None:
        all_vertices = set(vertices)
    else:
        all_vertices = set(parents.keys())
        for v, ps in parents.items():
            for p in ps:
                all_vertices.add(p)
        for o in outputs:
            all_vertices.add(o)

    if inputs is not None:
        input_vertices = set(inputs)
    else:
        input_vertices = set(v for v in all_vertices if not parents.get(v, []))

    # 2. Topological Layering
    layer = {}
    for v in input_vertices:
        layer[v] = 0

    children = defaultdict(list)
    for v, ps in parents.items():
        for p in ps:
            children[p].append(v)

    queue = deque(list(input_vertices))
    while queue:
        v = queue.popleft()
        for c in children[v]:
            if c not in layer:
                all_p_done = all(p in layer for p in parents.get(c, []))
                if all_p_done:
                    layer[c] = max(layer[p] for p in parents.get(c, [])) + 1
                    queue.append(c)

    for v in all_vertices:
        if v not in layer:
            layer[v] = 0

    layers = defaultdict(list)
    for v in all_vertices:
        layers[layer[v]].append(v)

    max_L = max(layers.keys()) if layers else 0

    # 3. Precompute Next Use Layer
    # next_use_layer[v] = min layer where v is a parent of a vertex in that layer
    # If v is never used as a parent, next_use_layer is max_L + 1
    next_use_layer = {v: max_L + 1 for v in all_vertices}
    for L in range(max_L + 1):
        for v in layers.get(L, []):
            for p in parents.get(v, []):
                if L < next_use_layer[p]:
                    next_use_layer[p] = L

    # 4. Greedy Simulation
    slow_mem = set(input_vertices)
    fast_mem = set()
    ops = []
    
    # Track stored status for computed vertices
    # If a vertex is in slow_mem, it is "stored" (blue).
    # Inputs are always in slow_mem initially.
    # Computed vertices are NOT in slow_mem until stored.

    def get_eviction_category(u, current_L):
        """
        Returns a tuple (category, next_use) for sorting.
        Lower category is better to evict.
        Category 0: Not needed later. (Delete is free, no future cost)
        Category 1: Needed later, is Input. (Delete is free, future Load is 1)
        Category 2: Needed later, is Computed. (Store is 1, future Load is 1)
        """
        nu = next_use_layer.get(u, max_L + 1)
        if nu > max_L:
            return (0, nu)
        
        if u in input_vertices:
            return (1, nu)
        else:
            return (2, nu)

    def evict_one(current_L):
        """Evict one vertex from fast_mem. Returns True if successful."""
        if not fast_mem:
            return False
            
        best_u = None
        best_key = None
        
        for u in fast_mem:
            cat, nu = get_eviction_category(u, current_L)
            # We want to evict the one with lowest category.
            # Within same category, evict the one with HIGHEST next_use (furthest away).
            key = (cat, -nu)
            if best_key is None or key < best_key:
                best_key = key
                best_u = u
        
        if best_u is None:
            return False

        cat, nu = get_eviction_category(best_u, current_L)
        
        if cat == 0:
            # Not needed, just delete
            ops.append(['delete', best_u])
        elif cat == 1:
            # Input, needed later. Delete (it's in slow, so we can reload)
            ops.append(['delete', best_u])
        elif cat == 2:
            # Computed, needed later. Must store to keep in slow
            ops.append(['store', best_u])
            slow_mem.add(best_u)
            
        fast_mem.discard(best_u)
        return True

    for L in range(max_L + 1):
        if time_check():
            break
            
        lv = layers.get(L, [])
        if not lv:
            continue

        # Identify needed parents for this layer
        needed_parents = set()
        for v in lv:
            if v in input_vertices:
                continue
            for p in parents.get(v, []):
                needed_parents.add(p)

        # Load needed parents
        # Sort parents by urgency (lowest next_use_layer first) to keep critical ones?
        # Actually, we just need to load them.
        for p in sorted(needed_parents, key=lambda x: next_use_layer.get(x, max_L + 1)):
            if p not in fast_mem:
                # Ensure budget
                while len(fast_mem) >= fast_memory_budget:
                    if not evict_one(L):
                        break
                    if time_check():
                        break
                
                if p not in fast_mem:
                    ops.append(['load', p])
                    fast_mem.add(p)
                    # Note: Loading an input does not remove it from slow_mem.
                    # Loading a computed vertex (if stored) does not remove it from slow_mem.
                    # slow_mem represents "available in slow memory".

        # Compute vertices in this layer
        for v in lv:
            if v in input_vertices:
                continue
            if time_check():
                break
            ops.append(['compute', v])
            fast_mem.add(v)
            # v is now in fast_mem. It is NOT in slow_mem (unless it was stored before, which is unlikely for new compute).
            # So v is "Red" in fast memory.

        # Proactive Cleanup: Delete vertices that are no longer needed
        # This frees up space without cost (delete is 0 transfers)
        to_delete = []
        for u in list(fast_mem):
            nu = next_use_layer.get(u, max_L + 1)
            if nu > max_L:
                to_delete.append(u)
                
        for u in to_delete:
            ops.append(['delete', u])
            fast_mem.discard(u)

    # Final Store for Outputs
    for o in outputs:
        if time_check():
            break
        if o in fast_mem:
            # If output is in fast, we must store it to finish "blue" in slow?
            # Prompt: "all outputs must finish blue". Blue usually means in slow memory or just computed?
            # "Inputs start blue... every red value consumes one fast-memory slot".
            # "Store" moves to slow.
            # If output is in fast_mem (red), we should store it.
            ops.append(['store', o])
            slow_mem.add(o)
            fast_mem.discard(o)
        # If output is already in slow_mem (e.g., it was an input, or stored earlier), it's fine.
        # If output was deleted and not stored? That's bad. But our logic stores computed vertices if needed.
        # Outputs are "needed" until the end?
        # next_use_layer for outputs is max_L+1 (no children).
        # So they fall into Category 0 (Not needed) in evict_one?
        # Wait. If an output is computed, and it has no children, next_use_layer is max_L+1.
        # So evict_one will see cat 0 and `delete` it.
        # This would lose the output!
        # We must treat Outputs specially.
        
    # Correction: Outputs must be stored.
    # We should modify get_eviction_category or the main loop to ensure outputs are not deleted.
    
    # Let's refine the logic:
    # An output vertex is "Needed" until the very end.
    # So for outputs, next_use_layer should be treated as "Current Layer" or "Infinite"?
    # Actually, once computed, an output is done. But it must be stored.
    # So, if an output is in fast_mem, we should store it immediately after computation?
    # Or keep it in fast_mem until the end?
    # If we keep it in fast_mem, it consumes a slot.
    # If we store it immediately, it frees the slot.
    # Storing is 1 transfer.
    # If we store it immediately, we pay 1.
    # If we keep it, we might evict it (if not careful) and lose it.
    # So, **Store Outputs Immediately** after computation.
    
    # Let's adjust the Compute Phase:
    # After computing v, if v is an output, store it immediately.
    
    # Re-implementing the loop with this fix.
    
    # Reset state
    slow_mem = set(input_vertices)
    fast_mem = set()
    ops = []
    
    def get_eviction_category_v2(u, current_L):
        nu = next_use_layer.get(u, max_L + 1)
        if u in output_set:
            # Outputs should be stored immediately, so they shouldn't be in fast_mem for long.
            # But if they are, they are critical.
            # Treat as Category 3 (Never evict if possible)?
            # Or just ensure we store them before they are evicted.
            pass
            
        if nu > max_L:
            # If it's an output, we MUST store it, not delete.
            if u in output_set:
                return (2, nu) # Force Store
            return (0, nu)
        
        if u in input_vertices:
            return (1, nu)
        else:
            return (2, nu)

    output_set = set(outputs)

    def evict_one_v2(current_L):
        if not fast_mem:
            return False
            
        best_u = None
        best_key = None
        
        for u in fast_mem:
            cat, nu = get_eviction_category_v2(u, current_L)
            key = (cat, -nu)
            if best_key is None or key < best_key:
                best_key = key
                best_u = u
        
        if best_u is None:
            return False

        cat, nu = get_eviction_category_v2(best_u, current_L)
        
        if cat == 0:
            ops.append(['delete', best_u])
        elif cat == 1:
            ops.append(['delete', best_u])
        elif cat == 2:
            ops.append(['store', best_u])
            slow_mem.add(best_u)
            
        fast_mem.discard(best_u)
        return True

    for L in range(max_L + 1):
        if time_check():
            break
            
        lv = layers.get(L, [])
        if not lv:
            continue

        needed_parents = set()
        for v in lv:
            if v in input_vertices:
                continue
            for p in parents.get(v, []):
                needed_parents.add(p)

        for p in sorted(needed_parents, key=lambda x: next_use_layer.get(x, max_L + 1)):
            if p not in fast_mem:
                while len(fast_mem) >= fast_memory_budget:
                    if not evict_one_v2(L):
                        break
                    if time_check():
                        break
                
                if p not in fast_mem:
                    ops.append(['load', p])
                    fast_mem.add(p)

        for v in lv:
            if v in input_vertices:
                continue
            if time_check():
                break
            ops.append(['compute', v])
            fast_mem.add(v)
            
            # If v is an output, store it immediately to free space and ensure it's blue
            if v in output_set:
                ops.append(['store', v])
                slow_mem.add(v)
                fast_mem.discard(v)

        # Proactive Cleanup for non-outputs
        to_delete = []
        for u in list(fast_mem):
            nu = next_use_layer.get(u, max_L + 1)
            if nu > max_L and u not in output_set:
                to_delete.append(u)
                
        for u in to_delete:
            ops.append(['delete', u])
            fast_mem.discard(u)

    # Final check for outputs
    for o in outputs:
        if o in fast_mem:
            ops.append(['store', o])
            slow_mem.add(o)
            fast_mem.discard(o)

    if len(ops) > 20000:
        ops = ops[:20000]

    return ops
# EVOLVE-BLOCK-END
