import numpy as np

def select_next_item(remaining_capacity, remaining_weights, remaining_values):
            n = len(remaining_weights)
            if n == 0:
                return -1
            
            # Ensure numpy arrays
            w = np.asarray(remaining_weights, dtype=float)
            v = np.asarray(remaining_values, dtype=float)
            C = float(remaining_capacity)
            
            # Feasible mask
            feasible = w <= C
            if not np.any(feasible):
                return -1
            
            # Get feasible indices and values
            idx = np.where(feasible)[0]
            wf = w[idx]
            vf = v[idx]
            
            # If only one item, return it
            if len(idx) == 1:
                return int(idx[0])
            
            # Calculate densities
            densities = vf / wf
            
            # Sort by density descending
            sorted_order = np.argsort(-densities)
            sorted_w = wf[sorted_order]
            sorted_v = vf[sorted_order]
            sorted_d = densities[sorted_order]
            
            # Find critical lambda
            cum_w = np.cumsum(sorted_w)
            
            # Find first index where cum_w > C
            # np.searchsorted(cum_w, C, side='left') gives index where C would be inserted
            # We want the first index where cum_w > C
            # If cum_w[-1] <= C, all fit. Lambda = 0? Or min density?
            # If all fit, we can just pick the best one?
            # Actually, if all fit, lambda is effectively 0 for the "excess" part, but the marginal utility is 0.
            # In that case, L = v. We pick max value.
            
            if cum_w[-1] <= C:
                # All items fit. Pick max value? Or max density?
                # If all fit, we should pick the one with max value to maximize total?
                # But we are picking ONE item now.
                # If all fit, the "shadow price" is 0.
                # L = v - 0*w = v.
                # So pick max value.
                max_val_idx_local = np.argmax(sorted_v)
                return int(idx[sorted_order[max_val_idx_local]])
            
            # Find critical index
            # cum_w is increasing.
            # We want the smallest k such that cum_w[k] > C
            critical_idx = np.searchsorted(cum_w, C, side='left')
            # Check if cum_w[critical_idx] <= C?
            # searchsorted with side='left' returns the index where C can be inserted to maintain order.
            # If cum_w = [2, 5, 9] and C=4.
            # searchsorted([2,5,9], 4, 'left') returns 1.
            # cum_w[1] = 5 > 4. Correct.
            # If C=2. searchsorted([2,5,9], 2, 'left') returns 0.
            # cum_w[0] = 2 <= 2.
            # In this case, the critical item is the one that *exceeds*?
            # If cum_w[k] <= C, then item k is fully included.
            # The critical item is the one that causes the overflow.
            # So if cum_w[critical_idx] <= C, we need critical_idx + 1?
            
            # Let's refine:
            # We want the density of the item that is partially filled.
            # That is the item at index k where cum_w[k-1] <= C < cum_w[k].
            
            if critical_idx == 0:
                # Even the first item exceeds C?
                # This implies sorted_w[0] > C?
                # But we filtered w <= C. So sorted_w[0] <= C.
                # So cum_w[0] <= C.
                # So critical_idx cannot be 0 in a way that cum_w[0] > C.
                # Wait, if C=1.5, w=[2,3]. Feasible: none.
                # We already handled feasible.
                # So sorted_w[0] <= C.
                # So cum_w[0] <= C.
                # So critical_idx will be at least 1 if there's an overflow later.
                # If cum_w[-1] <= C, we handled it.
                # So here, cum_w[-1] > C.
                # And cum_w[0] <= C.
                # So critical_idx is in [1, n-1].
                
                # Check:
                # If cum_w[critical_idx] <= C, then the overflow is at critical_idx + 1.
                if cum_w[critical_idx] <= C:
                    critical_idx += 1
            
            # Now critical_idx is the index of the item that overflows.
            # Lambda is the density of this item.
            lambda_val = sorted_d[critical_idx]
            
            # Calculate Lagrangian scores for all feasible items
            # L = v - lambda * w
            L = vf - lambda_val * wf
            
            # Pick max L
            best_local_idx = np.argmax(L)
            
            return int(idx[best_local_idx])
