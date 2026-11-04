            # ... inside construct ...
            top_candidates = [] # list of (gain, idx)
            
            for c_idx in candidates:
                gain = len(uncovered.intersection(block_to_triples[c_idx]))
                # Insert into top 5
                if len(top_candidates) < 5:
                    top_candidates.append((gain, c_idx))
                    top_candidates.sort(reverse=True)
                else:
                    if gain > top_candidates[-1][0]:
                        top_candidates.pop()
                        top_candidates.append((gain, c_idx))
                        top_candidates.sort(reverse=True)
            
            if top_candidates:
                # Pick randomly from the top 5
                # Weighted by gain? Or uniform?
                # Uniform among top 5 is a good heuristic
                chosen_idx = random.choice(top_candidates)[1]
