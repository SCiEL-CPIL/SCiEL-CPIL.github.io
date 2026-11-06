    def run_cse_on_bitmasks(bitmasks, num_inputs, seed, input_wire_ids):
        random.seed(seed)
        
        wire_masks = [1 << i for i in range(num_inputs)]
        wire_masks.append(0)
        
        mask_to_wire = {0: num_inputs}
        for i in range(num_inputs):
            mask_to_wire[1 << i] = i
            
        expressions = []
        for bm in bitmasks:
            expr = set()
            for i in range(num_inputs):
                if (bm >> i) & 1:
                    expr.add(i)
            expressions.append(expr)
            
        gates = []
        next_wire = num_inputs + 1
        
        while True:
            pair_count = defaultdict(int)
            
            for expr in expressions:
                if len(expr) < 2:
                    continue
                lst = list(expr)
                for i in range(len(lst)):
                    for j in range(i + 1, len(lst)):
                        u, v = lst[i], lst[j]
                        if u > v:
                            u, v = v, u
                        pair_count[(u, v)] += 1
                        
            if not pair_count:
                break
                
            max_count = max(pair_count.values())
            if max_count < 2:
                break
                
            candidates = [(pair, count) for pair, count in pair_count.items() if count == max_count]
            pair, _ = random.choice(candidates)
            u, v = pair
            
            new_mask = wire_masks[u] ^ wire_masks[v]
            
            if new_mask in mask_to_wire:
                existing_id = mask_to_wire[new_mask]
                for i in range(len(expressions)):
                    expr = expressions[i]
                    if u in expr and v in expr:
                        expr.remove(u)
                        expr.remove(v)
                        expr.add(existing_id)
                continue
            
            wire_masks.append(new_mask)
            mask_to_wire[new_mask] = next_wire
            new_id = next_wire
            next_wire += 1
            
            gates.append((u, v))
            
            for i in range(len(expressions)):
                expr = expressions[i]
                if u in expr and v in expr:
                    expr.remove(u)
                    expr.remove(v)
                    expr.add(new_id)
                    
        final_outputs = []
        for expr in expressions:
            if not expr:
                final_outputs.append(num_inputs)
            elif len(expr) == 1:
                final_outputs.append(list(expr)[0])
            else:
                target_mask = 0
                for w in expr:
                    target_mask ^= wire_masks[w]
                
                if target_mask in mask_to_wire:
                    final_outputs.append(mask_to_wire[target_mask])
                else:
                    lst = list(expr)
                    current = lst[0]
                    for w in lst[1:]:
                        gates.append((current, w))
                        new_mask = wire_masks[current] ^ wire_masks[w]
                        wire_masks.append(new_mask)
                        mask_to_wire[new_mask] = len(wire_masks) - 1
                        current = len(wire_masks) - 1
                    final_outputs.append(current)
                    
        return {
            'local_gates': gates,
            'local_outputs': final_outputs
        }
