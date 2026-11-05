<<<<<<< SEARCH
    def try_greedy(seed=None):
        """Run greedy synthesis with a given random seed for tie-breaking."""
        if seed is not None:
            random.seed(seed)
        
        gates = []
        next_wire = zero_wire + 1
        
        # Local copies
        rw = [set(s) for s in row_wires]
        ws = dict(wire_set)
        wd = dict(wire_depth)
        
        max_iter = 20000
        for _ in range(max_iter):
            if not elapsed_ok():
                break
            
            # Count co-occurrence of pairs in rows with size >= 2
            pair_count = defaultdict(int)
            for i in range(n_rows):
                rs = rw[i]
                if len(rs) < 2:
                    continue
                elems = list(rs)
                for x in range(len(elems)):
                    for y in range(x + 1, len(elems)):
                        a, b = elems[x], elems[y]
                        if a > b:
                            a, b = b, a
                        # Skip if either is zero wire
                        if a == zero_wire or b == zero_wire:
                            continue
                        # Skip if they represent the same set (XOR would be 0)
                        if ws[a] == ws[b]:
                            continue
                        pair_count[(a, b)] += 1
            
            if not pair_count:
                break
            
            # Find best pair with tie-breaking
            best_pair = None
            best_benefit = 0
            best_depth = 0
            for (a, b), cnt in pair_count.items():
                if cnt < 2:
                    continue
                benefit = cnt - 1
                # New depth would be max(depth[a], depth[b]) + 1
                new_depth = max(wd[a], wd[b]) + 1
                if new_depth > depth_limit:
                    continue
                if benefit > best_benefit or (benefit == best_benefit and new_depth < best_depth):
                    best_benefit = benefit
                    best_depth = new_depth
                    best_pair = (a, b)
            
            if best_pair is None:
                break
            
            a, b = best_pair
            new_wire = next_wire
            next_wire += 1
            
            # Compute the set for the new wire
            new_set = ws[a] ^ ws[b]
            if len(new_set) == 0:
                # This would produce zero, skip
                continue
            ws[new_wire] = new_set
            wd[new_wire] = max(wd[a], wd[b]) + 1
            gates.append([a, b])
            
            # Update all rows containing both a and b
            for i in range(n_rows):
                rs = rw[i]
                if a in rs and b in rs:
                    rs.discard(a)
                    rs.discard(b)
                    rs.add(new_wire)
        
        # Build outputs
        outputs = []
        for i in range(n_rows):
            rs = rw[i]
            if len(rs) == 0:
                outputs.append(zero_wire)
            elif len(rs) == 1:
                outputs.append(list(rs)[0])
            else:
                elems = list(rs)
                # Sort by depth to minimize overall depth
                elems.sort(key=lambda w: wd.get(w, 0))
                current = elems[0]
                for e in elems[1:]:
                    # Check depth
                    new_depth = max(wd.get(current, 0), wd.get(e, 0)) + 1
                    if new_depth > depth_limit:
                        # Can't add more gates due to depth, just output as is
                        break
                    gates.append([current, e])
                    current = next_wire
                    next_wire += 1
                    ws[current] = ws.get(current, frozenset())  # placeholder
                    wd[current] = new_depth
                outputs.append(current)
        
        return {'gates': gates, 'outputs': outputs}
=======
    def try_greedy(seed=None):
        """Run greedy synthesis with a given random seed for tie-breaking."""
        if seed is not None:
            random.seed(seed)
        
        gates = []
        next_wire = zero_wire + 1
        
        # Local copies
        rw = [set(s) for s in row_wires]
        ws = dict(wire_set)
        wd = dict(wire_depth)
        
        max_iter = 20000
        for _ in range(max_iter):
            if not elapsed_ok():
                break
            
            # Count co-occurrence of pairs in rows with size >= 2
            pair_count = defaultdict(int)
            for i in range(n_rows):
                rs = rw[i]
                if len(rs) < 2:
                    continue
                elems = list(rs)
                for x in range(len(elems)):
                    for y in range(x + 1, len(elems)):
                        a, b = elems[x], elems[y]
                        if a > b:
                            a, b = b, a
                        # Skip if either is zero wire
                        if a == zero_wire or b == zero_wire:
                            continue
                        # Skip if they represent the same set (XOR would be 0)
                        if ws[a] == ws[b]:
                            continue
                        pair_count[(a, b)] += 1
            
            if not pair_count:
                break
            
            # Find best pair with tie-breaking
            best_pair = None
            best_benefit = 0
            best_depth = 0
            for (a, b), cnt in pair_count.items():
                if cnt < 2:
                    continue
                benefit = cnt - 1
                # New depth would be max(depth[a], depth[b]) + 1
                new_depth = max(wd[a], wd[b]) + 1
                if new_depth > depth_limit:
                    continue
                if benefit > best_benefit or (benefit == best_benefit and new_depth < best_depth):
                    best_benefit = benefit
                    best_depth = new_depth
                    best_pair = (a, b)
            
            if best_pair is None:
                break
            
            a, b = best_pair
            new_wire = next_wire
            next_wire += 1
            
            # Compute the set for the new wire
            new_set = ws[a] ^ ws[b]
            if len(new_set) == 0:
                # This would produce zero, skip
                continue
            ws[new_wire] = new_set
            wd[new_wire] = max(wd[a], wd[b]) + 1
            gates.append([a, b])
            
            # Update all rows containing both a and b
            for i in range(n_rows):
                rs = rw[i]
                if a in rs and b in rs:
                    rs.discard(a)
                    rs.discard(b)
                    rs.add(new_wire)
        
        # Build outputs
        outputs = []
        for i in range(n_rows):
            rs = rw[i]
            if len(rs) == 0:
                outputs.append(zero_wire)
            elif len(rs) == 1:
                outputs.append(list(rs)[0])
            else:
                elems = list(rs)
                # Sort by depth to minimize overall depth
                elems.sort(key=lambda w: wd.get(w, 0))
                current = elems[0]
                for e in elems[1:]:
                    # Check depth
                    new_depth = max(wd.get(current, 0), wd.get(e, 0)) + 1
                    if new_depth > depth_limit:
                        # Can't add more gates due to depth, just output as is
                        break
                    gates.append([current, e])
                    current = next_wire
                    next_wire += 1
                    ws[current] = ws.get(current, frozenset())  # placeholder
                    wd[current] = new_depth
                outputs.append(current)
        
        # POLISH PASS: Remove redundant gates where output wire equals an existing wire
        # Build mapping from frozenset to wire index (prefer lowest index)
        set_to_wire = {}
        for w in range(zero_wire + 1, next_wire):
            s = ws.get(w, frozenset())
            if s not in set_to_wire:
                set_to_wire[s] = w
        
        # Also include input wires and zero wire
        for w in range(zero_wire + 1):
            s = ws.get(w, frozenset())
            if s not in set_to_wire:
                set_to_wire[s] = w
        
        # Identify redundant gates: gate k produces wire w = zero_wire+1+k
        # If ws[w] equals ws[other] for some other wire, we can potentially remove it
        # But we need to check if the gate's inputs are still needed elsewhere
        
        # For each gate, check if its output set matches an existing non-gate wire
        # or a gate that appears earlier
        redundant = [False] * len(gates)
        for k in range(len(gates)):
            w = zero_wire + 1 + k
            s = ws.get(w, frozenset())
            # Check if this set is already represented by a lower-index wire
            for other_w in range(w):
                if ws.get(other_w, frozenset()) == s:
                    redundant[k] = True
                    break
        
        # If any redundant gates, rebuild circuit without them
        if any(redundant):
            # Build new gates list and remap wires
            new_gates = []
            wire_map = {}  # old_wire -> new_wire
            next_new_wire = zero_wire + 1
            
            # Map all non-gate wires
            for w in range(zero_wire + 1):
                wire_map[w] = w
            
            for k in range(len(gates)):
                old_wire = zero_wire + 1 + k
                if redundant[k]:
                    # This gate is redundant; map its output to an equivalent wire
                    s = ws.get(old_wire, frozenset())
                    # Find the canonical wire for this set
                    canonical = None
                    for other_w in range(old_wire):
                        if ws.get(other_w, frozenset()) == s:
                            canonical = other_w
                            break
                    if canonical is not None:
                        wire_map[old_wire] = wire_map.get(canonical, canonical)
                    else:
                        wire_map[old_wire] = old_wire
                else:
                    new_gates.append([wire_map[gates[k][0]], wire_map[gates[k][1]]])
                    wire_map[old_wire] = next_new_wire
                    next_new_wire += 1
            
            # Remap outputs
            new_outputs = [wire_map.get(o, o) for o in outputs]
            
            # Verify correctness
            valid = True
            # Check that all wires in gates and outputs are valid
            max_wire = next_new_wire - 1
            for g in new_gates:
                if g[0] > max_wire or g[1] > max_wire:
                    valid = False
                    break
            if valid:
                for o in new_outputs:
                    if o > max_wire:
                        valid = False
                        break
            
            if valid:
                gates = new_gates
                outputs = new_outputs
        
        return {'gates': gates, 'outputs': outputs}
>>>>>>> REPLACE

