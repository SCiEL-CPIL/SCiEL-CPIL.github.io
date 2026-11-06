import math
def construct():
            import time
            start_time = time.time()
            deadline = 85 # 5s margin
            
            # 1. Define H and L domains
            H = list(range(129, 257))
            L_domain = list(range(2, 129))
            
            # 2. Precompute Gain(l)
            # Gain(l) = 1/l - sum(1/h for h in H if l divides h)
            gains = {}
            for l in L_domain:
                loss = 0.0
                for h in H:
                    if h % l == 0:
                        loss += 1.0 / h
                gains[l] = 1.0 / l - loss
                
            # Filter to positive gains
            candidates = [l for l in L_domain if gains[l] > 0]
            
            # 3. Build Conflict Graph for candidates
            # Map index to value
            n = len(candidates)
            conflict = [0] * n
            for i in range(n):
                li = candidates[i]
                for j in range(i+1, n):
                    lj = candidates[j]
                    # Check if lcm(li, lj) <= 256
                    # math.lcm
                    if math.lcm(li, lj) <= 256:
                        conflict[i] |= (1 << j)
                        conflict[j] |= (1 << i)
                        
            # 4. B&B
            best_sum = 0.0
            best_set = []
            
            # Sort candidates by gain desc for better pruning?
            # The order in bitmask matters for the bound if we use prefix sums.
            # Let's sort indices by gain desc.
            sorted_indices = sorted(range(n), key=lambda i: gains[candidates[i]], reverse=True)
            # Re-map candidates and conflicts to sorted order
            # This is getting complex. Let's stick to original order and just use the bound:
            # Bound = current_sum + sum(gains[i] for i in available if gains[i] > 0)
            
            def search(mask, current_sum, current_set):
                nonlocal best_sum, best_set
                if time.time() > deadline:
                    return
                
                # Calculate bound
                # Optimistic bound: add all positive gains from available nodes
                bound = current_sum
                m = mask
                while m:
                    lsb = m & -m
                    idx = lsb.bit_length() - 1
                    if gains[candidates[idx]] > 0:
                        bound += gains[candidates[idx]]
                    m -= lsb
                
                if bound <= best_sum + 1e-9:
                    return
                
                if mask == 0:
                    if current_sum > best_sum:
                        best_sum = current_sum
                        best_set = current_set[:]
                    return
                
                # Pick a node to branch on
                # Heuristic: pick the node with highest gain in mask?
                # Or just the first one?
                # Let's pick the one with highest gain to prune faster.
                m = mask
                best_idx = -1
                best_g = -1
                while m:
                    lsb = m & -m
                    idx = lsb.bit_length() - 1
                    if gains[candidates[idx]] > best_g:
                        best_g = gains[candidates[idx]]
                        best_idx = idx
                    m -= lsb
                    
                if best_idx == -1:
                    # No positive gain nodes left?
                    if current_sum > best_sum:
                        best_sum = current_sum
                        best_set = current_set[:]
                    return

                # Branch 1: Include best_idx
                li = candidates[best_idx]
                new_mask = mask & ~conflict[best_idx] & ~(1 << best_idx)
                current_set.append(li)
                search(new_mask, current_sum + gains[li], current_set)
                current_set.pop()
                
                # Branch 2: Exclude best_idx
                new_mask = mask & ~(1 << best_idx)
                search(new_mask, current_sum, current_set)
                
            # Initial call
            full_mask = (1 << n) - 1
            search(full_mask, 0.0, [])
            
            # 5. Construct final set
            L_star = best_set
            H_star = [h for h in H if all(h % l != 0 for l in L_star)]
            
            return L_star + H_star
