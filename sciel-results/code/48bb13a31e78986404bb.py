    def get_wire_for_set(target_set):
        if target_set in set_to_wire:
            return set_to_wire[target_set]
        
        elems = list(target_set)
        n = len(elems)
        
        if n == 0:
            w = zero_wire
            set_to_wire[target_set] = w
            return w
        if n == 1:
            w = elems[0]
            set_to_wire[target_set] = w
            return w
            
        best_cost = float('inf')
        best_a = None
        best_b = None
        
        # Try all splits if n is small
        if n <= 6:
            for mask in range(1, (1 << n) - 1):
                # Avoid symmetric
                if mask > ((1 << n) - 1) // 2:
                    continue
                a_set = frozenset(elems[i] for i in range(n) if mask & (1 << i))
                b_set = frozenset(target_set ^ a_set)
                
                # Estimate cost: 1 + cost(a) + cost(b)
                # We don't know cost yet, but we can estimate by size
                # Or just pick the split that minimizes max(size(a), size(b))
                # Which is balanced.
                
                # Let's just use balanced for now.
                pass
                
        # Balanced split
        mid = n // 2
        a_set = frozenset(elems[:mid])
        b_set = frozenset(elems[mid:])
        
        a_wire = get_wire_for_set(a_set)
        b_wire = get_wire_for_set(b_set)
        
        new_wire = next_wire
        next_wire += 1
        gates.append([a_wire, b_wire])
        wire_depth[new_wire] = max(wire_depth[a_wire], wire_depth[b_wire]) + 1
        set_to_wire[target_set] = new_wire
        
        return new_wire
