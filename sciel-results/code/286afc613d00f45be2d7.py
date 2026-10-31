    def local_search(S):
        S = set(S)
        in_S = [False] * N
        for v in S:
            in_S[v] = True
            
        improved = True
        max_iter = 100
        iter_count = 0
        
        while improved and iter_count < max_iter:
            improved = False
            iter_count += 1
            
            # Phase 1: Greedy Add
            # Try to add any vertex that has no neighbors in S
            for v in range(N):
                if not in_S[v]:
                    is_free = True
                    for u in adj[v]:
                        if in_S[u]:
                            is_free = False
                            break
                    if is_free:
                        in_S[v] = True
                        S.add(v)
                        improved = True
            if improved:
                continue
                
            # Phase 2: Kempe Chain
            visited = [False] * N
            for start in range(N):
                if in_S[start] or visited[start]:
                    continue
                    
                # Find component C in V\S
                C = []
                queue = [start]
                visited[start] = True
                R = set()
                
                while queue:
                    u = queue.pop(0)
                    C.append(u)
                    for w in adj[u]:
                        if in_S[w]:
                            R.add(w)
                        else:
                            if not visited[w]:
                                visited[w] = True
                                queue.append(w)
                                
                if len(C) > len(R):
                    for v in R:
                        in_S[v] = False
                        S.remove(v)
                    for v in C:
                        in_S[v] = True
                        S.add(v)
                    improved = True
                    break # Restart loop
            
            if improved:
                continue
                
            # Phase 3: Random Perturbation (Escape Local Minima)
            if S:
                v_remove = random.choice(list(S))
                in_S[v_remove] = False
                S.remove(v_remove)
                # We don't set improved=True here necessarily, 
                # but we want to try to fix it in the next loop.
                # However, if we just remove, the set size decreases.
                # The next "Add" phase might add it back or a neighbor.
                # To ensure progress, we can try to add a neighbor of v_remove immediately?
                # No, let the loop handle it.
                # But if we remove and the next add phase adds it back, we are stuck in a loop.
                # To avoid infinite loop of remove/add same vertex:
                # We can mark v_remove as "temporarily banned" or just accept that 
                # random perturbations might fail. 
                # A better perturbation: Remove v_remove, and *force* add a random neighbor 
                # that is now free? No, that's complex.
                # Simple approach: If we remove and add back, we waste an iteration.
                # To prevent this, we can keep a history or just rely on max_iter.
                # Given max_iter=100, it's fine.
                improved = True # Force re-evaluation
            else:
                break
                
        return S
