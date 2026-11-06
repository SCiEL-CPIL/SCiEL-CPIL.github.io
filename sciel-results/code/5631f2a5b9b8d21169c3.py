    while time.time() < deadline:
        S = set()
        s_list = []
        candidates = list(range(1, N + 1))
        random.shuffle(candidates)
        
        for idx, x in enumerate(candidates):
            # Pruning: even if we add all remaining, we won't beat best
            if len(s_list) + (len(candidates) - idx) <= len(best_set):
                break
            
            can_add = True
            for i in range(len(s_list)):
                a = s_list[i]
                for j in range(i + 1, len(s_list)):
                    b = s_list[j]
                    if gcds[a][b] == gcds[a][x] == gcds[b][x]:
                        can_add = False
                        break
                if not can_add:
                    break
            
            if can_add:
                S.add(x)
                s_list.append(x)
        
        if len(S) > len(best_set):
            best_set = S
