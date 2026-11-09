def construct():
                N = 256
                S = set()
                # Greedy
                for x in range(1, N+1):
                    valid = True
                    # Since x is largest, only check x = 2b - a
                    for a in S:
                        for b in S:
                            if a < b and 2*b - a == x:
                                valid = False
                                break
                        if not valid: break
                    if valid:
                        S.add(x)
                
                # Local Search
                # Precompute reciprocals
                recips = {i: 1.0/i for i in range(1, N+1)}
                
                def is_valid_add(y, S_set):
                    if y in S_set:
                        return False
                    for a in S_set:
                        # Case 1: y, a, b -> b = 2a - y
                        b = 2*a - y
                        if b in S_set and b != y:
                            return False
                        # Case 2: a, y, b -> b = 2y - a
                        b = 2*y - a
                        if b in S_set and b != y:
                            return False
                        # Case 3: a, b, y -> a = 2b - y ? No, a is in S_set.
                        # If a, b, y is AP, then b = (a+y)/2.
                        # So check if (a+y) is even and (a+y)//2 in S_set.
                        s = a + y
                        if s % 2 == 0:
                            b = s // 2
                            if b in S_set and b != y:
                                return False
                    return True

                improved = True
                while improved:
                    improved = False
                    current_sum = sum(recips[x] for x in S)
                    for x in list(S):
                        S_trial = S - {x}
                        sum_trial = current_sum - recips[x]
                        
                        # Add valid y
                        for y in range(1, N+1):
                            if y in S_trial:
                                continue
                            if is_valid_add(y, S_trial):
                                S_trial.add(y)
                                sum_trial += recips[y]
                        
                        if sum_trial > current_sum:
                            S = S_trial
                            current_sum = sum_trial
                            improved = True
                            break
                            
                return list(S)
