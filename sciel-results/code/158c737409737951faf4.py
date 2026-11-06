                V_rand = list(V)
                rng_restart.shuffle(V_rand)
                S_rand = []
                S_rand_set = set()
                S_rand_sums = set()
                S_rand_diffs = set()
                for x in V_rand:
                    if 2 * x in S_rand_set:
                        continue
                    if x in S_rand_diffs:
                        continue
                    if x in S_rand_sums:
                        continue
                    
                    for a in S_rand:
                        S_rand_sums.add(x + a)
                        S_rand_diffs.add(x - a)
                        S_rand_diffs.add(a - x)
                    S_rand_sums.add(2 * x)
                    S_rand_diffs.add(0)
                    
                    S_rand.append(x)
                    S_rand_set.add(x)
