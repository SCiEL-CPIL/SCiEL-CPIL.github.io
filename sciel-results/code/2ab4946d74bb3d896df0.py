<<<<<<< SEARCH
                if max_gain == 0:
                    # No gain, pick random point not in B
                    candidates = [x for x in range(v) if x not in B]
                    if not candidates:
                        break
                    x = rng.choice(candidates)
                else:
                    # Pick random among max gain candidates
                    max_candidates = np.where(gains == max_gain)[0]
                    max_candidates = [x for x in max_candidates if x not in B]
                    if not max_candidates:
                         # Fallback
                         candidates = [x for x in range(v) if x not in B]
                         x = rng.choice(candidates)
                    else:
                        x = rng.choice(max_candidates)
                
                B.append(x)
=======
                if max_gain == 0:
                    # No gain, pick random point not in B
                    candidates = [x for x in range(v) if x not in B]
                    if not candidates:
                        break
                    x = int(rng.choice(candidates))
                else:
                    # Pick random among max gain candidates
                    max_candidates = np.where(gains == max_gain)[0]
                    max_candidates = [int(x) for x in max_candidates if x not in B]
                    if not max_candidates:
                         # Fallback
                         candidates = [x for x in range(v) if x not in B]
                         x = int(rng.choice(candidates))
                    else:
                        x = int(rng.choice(max_candidates))
                
                B.append(x)
>>>>>>> REPLACE
