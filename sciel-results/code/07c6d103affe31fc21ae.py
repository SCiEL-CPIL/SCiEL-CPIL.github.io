<<<<<<< SEARCH
    def local_search(selected, time_limit, max_rounds=50):
        """Try swap operations: replace one block with another to potentially remove a third."""
        if len(selected) < 3:
            return selected
        
        selected_list = list(selected)
        cover_counts = np.zeros(num_triples, dtype=np.int32)
        for b in selected_list:
            cover_counts[block_triple_indices[b]] += 1
        
        best_selected = selected_list[:]
        best_len = len(selected_list)
        
        for _ in range(max_rounds):
            if time.time() > time_limit:
                break
            
            improved = False
            random.shuffle(selected_list)
            
            for i, A in enumerate(selected_list):
                if time.time() > time_limit:
                    break
                
                A_triples = block_triple_indices[A]
                
                # Find triples only covered by A in current selection
                unique_mask = (cover_counts[A_triples] == 1)
                if not np.any(unique_mask):
                    # A is redundant, remove it
                    cover_counts[A_triples] -= 1
                    selected_list.pop(i)
                    best_len -= 1
                    best_selected = selected_list[:]
                    improved = True
                    break
                
                unique_triples = A_triples[unique_mask]
                # Find candidate B's that cover all unique triples of A
                candidate_Bs = set(triple_blocks[unique_triples[0]])
                for ut in unique_triples[1:]:
                    candidate_Bs &= set(triple_blocks[ut])
                
                candidate_Bs.discard(A)
                
                # Sort candidates by potential (fewer unique triples they'd need to cover)
                # Just try them in random order for diversity
                cand_list = list(candidate_Bs)
                random.shuffle(cand_list)
                
                for B in cand_list:
                    if B in selected_list:
                        continue
                    
                    # Try: remove A, add B
                    cover_counts[A_triples] -= 1
                    cover_counts[block_triple_indices[B]] += 1
                    
                    # Check if we can remove some C - find ALL removable C's
                    removable_Cs = []
                    for j, C in enumerate(selected_list):
                        if C == A:
                            continue
                        C_triples = block_triple_indices[C]
                        if np.all(cover_counts[C_triples] > 1):
                            removable_Cs.append(j)
                    
                    if removable_Cs:
                        # Pick the first one (could try to be smarter here)
                        removed_C = removable_Cs[0]
                        cover_counts[block_triple_indices[removed_C]] -= 1
                        # Rebuild selected_list
                        new_sel = []
                        for idx, b in enumerate(selected_list):
                            if idx != removed_C and b != A:
                                new_sel.append(b)
                        new_sel.append(B)
                        selected_list = new_sel
                        best_len -= 1
                        best_selected = selected_list[:]
                        improved = True
                        break
                    else:
                        # Revert
                        cover_counts[A_triples] += 1
                        cover_counts[block_triple_indices[B]] -= 1
            
            if not improved:
                break
        
        return best_selected
=======
    def local_search(selected, time_limit, max_rounds=50):
        """Try swap operations: replace one block with another to potentially remove a third."""
        if len(selected) < 3:
            return selected
        
        selected_list = list(selected)
        selected_set = set(selected_list)
        cover_counts = np.zeros(num_triples, dtype=np.int32)
        for b in selected_list:
            cover_counts[block_triple_indices[b]] += 1
        
        best_selected = selected_list[:]
        best_len = len(selected_list)
        
        for _ in range(max_rounds):
            if time.time() > time_limit:
                break
            
            improved = False
            random.shuffle(selected_list)
            
            for i, A in enumerate(selected_list):
                if time.time() > time_limit:
                    break
                
                A_triples = block_triple_indices[A]
                
                # Find triples only covered by A in current selection
                unique_mask = (cover_counts[A_triples] == 1)
                if not np.any(unique_mask):
                    # A is redundant, remove it
                    cover_counts[A_triples] -= 1
                    selected_list.pop(i)
                    selected_set.discard(A)
                    best_len -= 1
                    best_selected = selected_list[:]
                    improved = True
                    break
                
                unique_triples = A_triples[unique_mask]
                # Find candidate B's that cover all unique triples of A
                candidate_Bs = set(triple_blocks[unique_triples[0]])
                for ut in unique_triples[1:]:
                    candidate_Bs &= set(triple_blocks[ut])
                
                candidate_Bs.discard(A)
                
                # Score candidates by how many currently-selected blocks they overlap with
                # (more overlap = more likely to enable removal)
                cand_list = list(candidate_Bs - selected_set)
                if not cand_list:
                    continue
                
                # Score: prefer blocks that cover many triples already covered by selected blocks
                # This increases chance of making some block redundant
                def score_candidate(B):
                    B_triples = block_triple_indices[B]
                    # Count how many of B's triples are covered 2+ times (excluding A's contribution)
                    # After removing A, these would be at count >= 2
                    overlap_with_selected = 0
                    for t_idx in B_triples:
                        if cover_counts[t_idx] >= 2:
                            overlap_with_selected += 1
                    return overlap_with_selected
                
                cand_list.sort(key=score_candidate, reverse=True)
                random.shuffle(cand_list[:min(20, len(cand_list))])  # Shuffle top candidates for diversity
                cand_list = cand_list[:50]  # Limit candidates tried per A
                
                for B in cand_list:
                    if B in selected_set:
                        continue
                    
                    # Try: remove A, add B
                    cover_counts[A_triples] -= 1
                    B_triples = block_triple_indices[B]
                    cover_counts[B_triples] += 1
                    
                    # Check if we can remove some C - find ALL removable C's
                    removable_Cs = []
                    for j, C in enumerate(selected_list):
                        if C == A:
                            continue
                        C_triples = block_triple_indices[C]
                        if np.all(cover_counts[C_triples] > 1):
                            removable_Cs.append(j)
                    
                    if removable_Cs:
                        # Pick the C that covers the most unique triples (hardest to replace)
                        # Actually, pick any removable one - prefer one with most redundancy
                        best_C_idx = max(removable_Cs, key=lambda j: np.sum(cover_counts[block_triple_indices[selected_list[j]]] > 2))
                        removed_C = selected_list[best_C_idx]
                        cover_counts[block_triple_indices[removed_C]] -= 1
                        # Rebuild selected_list
                        new_sel = []
                        for idx, b in enumerate(selected_list):
                            if idx != best_C_idx and b != A:
                                new_sel.append(b)
                        new_sel.append(B)
                        selected_list = new_sel
                        selected_set = set(selected_list)
                        best_len -= 1
                        best_selected = selected_list[:]
                        improved = True
                        break
                    else:
                        # Revert
                        cover_counts[A_triples] += 1
                        cover_counts[B_triples] -= 1
            
            if not improved:
                break
        
        return best_selected
>>>>>>> REPLACE

