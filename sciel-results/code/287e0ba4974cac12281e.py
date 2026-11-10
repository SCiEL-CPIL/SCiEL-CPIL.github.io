    def local_search(arr: np.ndarray, skipped: np.ndarray) -> np.ndarray:
        if len(skipped) == 0:
            return arr
        
        S = [tuple(p) for p in arr]
        C = [tuple(p) for p in skipped]
        
        improved = True
        swap_attempts = 0
        max_swap_attempts = 2000 # Limit per local search call
        
        while improved and time.time() < deadline and swap_attempts < max_swap_attempts:
            improved = False
            swap_attempts += 1
            
            # 1. Try to add any point from C
            new_C = []
            S_arr = np.array(S, dtype=np.int64)
            S_sq = (S_arr ** 2).sum(axis=1)
            for p in C:
                if can_add(S_arr, np.array(p, dtype=np.int64), S_sq):
                    S.append(p)
                    S_arr = np.vstack([S_arr, np.array(p, dtype=np.int64)])
                    S_sq = np.append(S_sq, (p ** 2).sum())
                    improved = True
                else:
                    new_C.append(p)
            C = new_C
            
            if not C:
                break
            
            # 2. If stuck, try a random swap
            if not improved:
                if len(S) > 0 and len(C) > 0:
                    i = np.random.randint(len(S))
                    j = np.random.randint(len(C))
                    p_old = S[i]
                    p_new = C[j]
                    
                    # Check if p_new can be added to S \ {p_old}
                    # This is equivalent to checking if S' is valid
                    if i == 0:
                        reduced_S = S[1:]
                    else:
                        reduced_S = S[:i] + S[i+1:]
                    
                    if reduced_S:
                        reduced_S_arr = np.array(reduced_S, dtype=np.int64)
                        reduced_S_sq = (reduced_S_arr ** 2).sum(axis=1)
                        if can_add(reduced_S_arr, np.array(p_new, dtype=np.int64), reduced_S_sq):
                            # Valid swap
                            S[i] = p_new
                            C[j] = p_old
                            improved = True # Try to add more
                    else:
                        # S had only 1 point, so reduced_S is empty, always valid
                        S[i] = p_new
                        C[j] = p_old
                        improved = True
        
        return np.array(S, dtype=np.int64)
