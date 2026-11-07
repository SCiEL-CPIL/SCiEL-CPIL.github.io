            # Sort candidates by degree in available graph
            def candidate_score(h):
                count = 0
                for nb_h in range(n):
                    if nb_h == h:
                        continue
                    edge = (min(h, nb_h), max(h, nb_h))
                    if edge in available:
                        count += 1
                return -count
            
            candidates.sort(key=candidate_score)
            
            # Randomize slightly to explore different branches
            # If we always pick the best, we might get stuck.
            # Let's shuffle the top few or use rng to pick a random one from the top half?
            # Simpler: with probability 0.5, shuffle the entire list.
            # Or just shuffle if we are in a "random" mode.
            # Let's shuffle with probability 0.3
            if rng.random() < 0.3:
                rng.shuffle(candidates)
