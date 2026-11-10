    def local_search(initial_S, rng):
        S = set(initial_S)
        best_S = set(S)
        best_size = len(S)
        stall_count = 0
        
        while time.time() < deadline:
            # Phase 1: Try to add any vertex
            added_any = False
            candidates = [v for v in range(N) if v not in S]
            rng.shuffle(candidates)
            
            for v in candidates:
                if can_add(v, S):
                    S.add(v)
                    added_any = True
            
            if len(S) > best_size:
                best_size = len(S)
                best_S = set(S)
                stall_count = 0 # Reset stall count on improvement
            else:
                stall_count += 1
            
            if added_any:
                continue
            
            # Phase 2: No vertex can be added. Kick.
            if not S:
                break
            
            # Increase kick size if stalled
            if stall_count > 10:
                k = rng.randint(2, 5)
            else:
                k = rng.randint(1, 2)
                
            if k > len(S):
                k = len(S)
            
            to_remove = rng.sample(list(S), k)
            for v in to_remove:
                S.remove(v)
            
            # If we removed everything, restart phase 1
            # If S is empty, we will just rebuild from scratch in next iteration?
            # Actually, if S is empty, candidates is all N, and we will add greedily.
            
        return best_S
