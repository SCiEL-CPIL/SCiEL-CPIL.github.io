<<<<<<< SEARCH
    # Local search: try to add elements, or swap to gain size
    if time.time() < deadline - 3:
        # First, try pure additions
        for _ in range(500):
            if time.time() > deadline:
                break
            # Try adding a random element
            x = np.random.randint(1, N+1)
            if best_indicator[x] == 0:
                valid, new_counts = add_element_check(best_counts, best_indicator, x)
                if valid:
                    best_set.append(x)
                    best_size += 1
                    best_counts = new_counts
                    best_indicator[x] = 1
            else:
                # Try swap: remove one element, add this one
                idx = np.random.randint(len(best_set))
                removed = best_set[idx]
                # Remove from counts
                temp_indicator = best_indicator.copy()
                temp_indicator[removed] = 0
                # Recompute counts after removal
                f = np.zeros(fft_size)
                f[1:N+1] = temp_indicator[1:N+1].astype(float)
                F = np.fft.fft(f)
                conv = np.fft.ifft(F * F).real
                all_sums = np.arange(2, 2*N+1)
                mids = all_sums // 2
                mid_valid = (mids >= 1) & (mids <= N)
                mid_in_set = np.zeros(2*N+1, dtype=np.float64)
                mid_in_set[1:N+1] = temp_indicator[1:N+1].astype(float)
                diag = np.where(mid_valid, mid_in_set[mids], 0.0)
                temp_counts = (conv[2:2*N+1] + diag) / 2.0
                
                valid, new_counts = add_element_check(temp_counts, temp_indicator, x)
                if valid:
                    new_set = best_set[:idx] + best_set[idx+1:] + [x]
                    best_set = new_set
                    best_size = len(best_set)
                    best_counts = new_counts
                    best_indicator = temp_indicator
                    best_indicator[x] = 1
                # If not valid, keep original
=======
    # Local search: try to add elements, or swap to gain size
    if time.time() < deadline - 3:
        # First, try pure additions using spectral guidance
        remaining = set(range(1, N+1)) - set(best_set)
        for _ in range(500):
            if time.time() > deadline:
                break
            if not remaining:
                break
            # Try adding an element guided by spectral score
            if len(remaining) > 30:
                sample = np.random.choice(list(remaining), size=30, replace=False)
            else:
                sample = list(remaining)
            
            best_x = None
            best_score = float('inf')
            for x in sample:
                score = spectral_score(best_counts, best_indicator, x)
                if score < best_score:
                    best_score = score
                    best_x = x
            
            if best_x is not None and best_score != float('inf'):
                valid, new_counts = add_element_check(best_counts, best_indicator, best_x)
                if valid:
                    best_set.append(best_x)
                    best_size += 1
                    best_counts = new_counts
                    best_indicator[best_x] = 1
                    remaining.remove(best_x)
            else:
                # Fallback: try random
                x = np.random.choice(list(remaining))
                valid, new_counts = add_element_check(best_counts, best_indicator, x)
                if valid:
                    best_set.append(x)
                    best_size += 1
                    best_counts = new_counts
                    best_indicator[x] = 1
                    remaining.remove(x)
        
        # Try swaps to improve size or quality
        if time.time() < deadline - 1:
            for _ in range(200):
                if time.time() > deadline:
                    break
                if not remaining:
                    break
                
                # Try swap: remove one element, add this one
                idx = np.random.randint(len(best_set))
                removed = best_set[idx]
                
                # Recompute counts after removal
                temp_indicator = best_indicator.copy()
                temp_indicator[removed] = 0
                f = np.zeros(fft_size)
                f[1:N+1] = temp_indicator[1:N+1].astype(float)
                F = np.fft.fft(f)
                conv = np.fft.ifft(F * F).real
                all_sums = np.arange(2, 2*N+1)
                mids = all_sums // 2
                mid_valid = (mids >= 1) & (mids <= N)
                mid_in_set = np.zeros(2*N+1, dtype=np.float64)
                mid_in_set[1:N+1] = temp_indicator[1:N+1].astype(float)
                diag = np.where(mid_valid, mid_in_set[mids], 0.0)
                temp_counts = (conv[2:2*N+1] + diag) / 2.0
                
                # Find best replacement using spectral score
                best_x = None
                best_score = float('inf')
                if len(remaining) > 30:
                    sample = np.random.choice(list(remaining), size=30, replace=False)
                else:
                    sample = list(remaining)
                
                for x in sample:
                    score = spectral_score(temp_counts, temp_indicator, x)
                    if score < best_score:
                        best_score = score
                        best_x = x
                
                if best_x is not None and best_score != float('inf'):
                    valid, new_counts = add_element_check(temp_counts, temp_indicator, best_x)
                    if valid:
                        new_set = best_set[:idx] + best_set[idx+1:] + [best_x]
                        best_set = new_set
                        best_size = len(best_set)
                        best_counts = new_counts
                        best_indicator = temp_indicator
                        best_indicator[best_x] = 1
                        remaining.remove(best_x)
>>>>>>> REPLACE

