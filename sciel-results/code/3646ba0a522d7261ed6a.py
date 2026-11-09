<<<<<<< SEARCH
                            # Find an available host for neighbor
                            candidates = []
                            for h in range(20):
                                if h in used_hosts:
                                    continue
                                edge = (min(mapping[node], h), max(mapping[node], h))
                                if edge not in used_edges:
                                    candidates.append(h)
=======
                            # Find an available host for neighbor
                            # Optimization: Iterate over available hosts
                            candidates = []
                            for h in available_hosts:
                                edge = (min(mapping[node], h), max(mapping[node], h))
                                if edge not in used_edges:
                                    candidates.append(h)
>>>>>>> REPLACE
