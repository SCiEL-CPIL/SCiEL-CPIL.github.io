        def local_search(code):
            code_set = set(code)
            covered_mask = 0
            for c in code_set:
                covered_mask |= masks[c]
                
            improved = True
            while improved:
                improved = False
                
                # 1. Prune
                for c in list(code_set):
                    # If c is redundant, removing it doesn't change covered_mask
                    # i.e., covered_mask & ~masks[c] == covered_mask
                    if (covered_mask & ~masks[c]) == covered_mask:
                        code_set.remove(c)
                        covered_mask &= ~masks[c]
                        improved = True
                        break
                        
                if improved:
                    continue

                # 2. Swap
                swapped = False
                code_list = list(code_set)
                random.shuffle(code_list)
                
                for c in code_list:
                    # Calculate private words of c
                    others_mask = covered_mask & ~masks[c]
                    private = masks[c] & ~others_mask
                    
                    if private == 0:
                        # Redundant (should be caught by prune, but safe)
                        code_set.remove(c)
                        covered_mask &= ~masks[c]
                        improved = True
                        swapped = True
                        break
                        
                    # Find replacement w
                    # w must cover all private words
                    # (private & ~masks[w]) == 0
                    found_w = -1
                    for w in range(total_words):
                        if w in code_set:
                            continue
                        if (private & ~masks[w]) == 0:
                            found_w = w
                            break
                            
                    if found_w != -1:
                        code_set.remove(c)
                        code_set.add(found_w)
                        # Recalculate covered_mask
                        covered_mask = 0
                        for o in code_set:
                            covered_mask |= masks[o]
                        improved = True
                        swapped = True
                        break
                        
                if swapped:
                    continue
                    
            return code_set
