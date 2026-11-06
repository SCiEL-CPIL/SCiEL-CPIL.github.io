<<<<<<< SEARCH
    def naive_schedule():
        """Generate a naive but valid schedule."""
        ops = []
        # Load all inputs
        for v in inputs:
            ops.append(['load', v])
        
        # Compute all non-input vertices in topological order
        computed = set(inputs)
        for v in topo_order:
            if v in input_set:
                continue
            # All parents should be computed
            for p in parents_map.get(v, []):
                if p not in computed:
                    # This shouldn't happen in valid topo order
                    pass
            ops.append(['compute', v])
            computed.add(v)
        
        # Store all outputs
        for v in outputs:
            ops.append(['store', v])
        
        return ops
=======
    def naive_schedule():
        """Generate a naive but valid schedule.
        Strategy: Load all inputs. Compute vertices in topo order.
        Keep all computed values in fast memory (if budget allows) or store/delete aggressively.
        Since this is a fallback, we prioritize validity over optimality.
        If fast_memory_budget is small, we might need to delete.
        However, 'load' puts in fast. 'compute' puts in fast.
        'store' moves to slow. 'delete' removes.
        
        Simplest valid schedule that respects memory:
        1. Load all inputs.
        2. For each vertex in topo order:
           - Ensure parents are in fast memory (load if in slow).
           - Compute vertex.
           - If fast memory is full, store/delete a non-critical vertex.
           - If vertex is output, store it at the end? No, store when done?
           - Actually, we must finish with outputs in slow memory (blue).
           
        Let's try a simple "load all, compute all, store all" if budget is large enough.
        If budget is small, we need a smarter fallback.
        
        Fallback Strategy:
        1. Load all inputs.
        2. Iterate topo order.
        3. For each v:
           a. For each parent p: if p not in fast, load p (if in slow).
           b. Compute v.
           c. If v is an output, we will store it later.
           d. If len(fast) > budget:
              - Find a vertex to store/delete. Prefer storing intermediates that are no longer needed.
              - If no such vertex, store/delete the one with least future use.
        4. At the end, store all outputs.
        """
        ops = []
        in_fast = set()
        in_slow = set(inputs) # Inputs start in slow (blue)
        computed = set()
        
        # 1. Load all inputs
        for v in inputs:
            ops.append(['load', v])
            in_fast.add(v)
            in_slow.discard(v)
            
        # 2. Process topo order
        for v in topo_order:
            if v in input_set:
                continue # Already loaded
                
            # Ensure parents are in fast
            for p in parents_map.get(v, []):
                if p not in in_fast:
                    if p in in_slow:
                        ops.append(['load', p])
                        in_fast.add(p)
                        in_slow.discard(p)
                    else:
                        # Parent not available? This would be an error in DAG or previous steps.
                        # For a valid naive schedule, this shouldn't happen if topo order is correct.
                        pass
            
            # Compute v
            ops.append(['compute', v])
            in_fast.add(v)
            computed.add(v)
            
            # Memory management
            while len(in_fast) > fast_memory_budget:
                # Find a candidate to evict
                # Prefer non-outputs, non-parents-of-future-vertices
                evicted = False
                for candidate in list(in_fast):
                    # Don't evict v just computed? Maybe okay if not needed immediately.
                    # But let's try to evict others first.
                    if candidate == v:
                        continue
                    # Check if candidate is needed by any future vertex
                    needed = False
                    for c in children_map.get(candidate, []):
                        if c not in computed and c not in in_fast:
                            # c is future
                            needed = True
                            break
                    if not needed and candidate not in output_set:
                        # Safe to evict
                        ops.append(['store', candidate])
                        in_fast.discard(candidate)
                        in_slow.add(candidate)
                        evicted = True
                        break
                
                if not evicted:
                    # Must evict something. Pick one.
                    # Prefer non-output.
                    candidates = [c for c in in_fast if c not in output_set]
                    if not candidates:
                        candidates = list(in_fast)
                    if candidates:
                        # Pick the one with least future usage
                        min_use = float('inf')
                        best_c = candidates[0]
                        for c in candidates:
                            use = len([ch for ch in children_map.get(c, []) if ch not in computed])
                            if use < min_use:
                                min_use = use
                                best_c = c
                        ops.append(['store', best_c])
                        in_fast.discard(best_c)
                        in_slow.add(best_c)
        
        # 3. Store all outputs
        for v in outputs:
            if v in in_fast:
                ops.append(['store', v])
                in_fast.discard(v)
                in_slow.add(v)
            # If v is already in slow, it's fine.
            
        return ops
>>>>>>> REPLACE

