            # If no swap found, try random perturbation
            if not improved and time.time() < deadline - 10:
                # Pick a random word to add
                rand_w = random.randint(0, total_words - 1)
                if rand_w not in code:
                    old_code = code.copy()
                    old_counts = counts.copy()
                    code.append(rand_w)
                    counts[balls[rand_w]] += 1
                    
                    # Prune aggressively
                    prune_changed = True
                    while prune_changed:
                        prune_changed = False
                        for i in range(len(code) - 1, -1, -1):
                            c = code[i]
                            if np.min(counts[balls[c]]) > 1:
                                code.pop(i)
                                counts[balls[c]] -= 1
                                prune_changed = True
                                break
                    
                    if len(code) < len(old_code):
                        improved = True
                        # Keep the new code
                    else:
                        # Revert
                        code = old_code
                        counts = old_counts
