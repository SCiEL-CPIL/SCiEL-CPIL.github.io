                added_ys = []
                for y in range(1, N + 1):
                    if y in S_set:
                        continue
                    if can_add(y, S_set):
                        S_set.add(y)
                        added_ys.append(y)
                        W_temp += recips[y]
                        added_any = True
                
                if added_any and W_temp > W:
                    S = list(S_set)
                    W = W_temp
                    improved = True
                    break
                else:
                    # Revert
                    for y in added_ys:
                        S_set.remove(y)
                    S_set.add(x)
