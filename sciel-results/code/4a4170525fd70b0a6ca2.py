   available_cand = np.ones(len(cand_indices), dtype=bool)
   added = []
   # We need to check conflicts between candidates.
   # We can maintain a mask of "blocked" candidates.
   # Initially none.
   # When we pick u, we block all candidates v such that conflict[u, v] is True.
   
   # To do this efficiently without submatrix:
   # We have cand_indices.
   # We need to check conflict[u, cand_indices] for each u.
   
   # This is still O(|C|^2).
   
   # Let's stick to the submatrix but add a size check.
   if len(cand_indices) > 3000:
       # Use a simpler heuristic: Random greedy
       # Pick random candidate, if compatible with added, add.
       # This is O(|C| * |Added|).
       random_cand = cand_indices[np.random.permutation(len(cand_indices))]
       for c in random_cand:
           # Check if c is compatible with all added
           # added is a list.
           # If len(added) is small, this is fast.
           if len(added) == 0:
               added.append(c)
               continue
           # Check conflict[c, added]
           # conflict[c, added] is a boolean array
           if not np.any(conflict[c, added]):
               added.append(c)
   else:
       # Use Min-Degree Greedy with submatrix
       conflict_sub = conflict[np.ix_(cand_indices, cand_indices)]
       degrees_sub = conflict_sub.sum(axis=1)
       available_sub = np.ones(len(cand_indices), dtype=bool)
       while np.any(available_sub):
           avail_idx = np.where(available_sub)[0]
           if len(avail_idx) == 0: break
           min_deg_idx = np.argmin(degrees_sub[avail_idx])
           chosen_local = avail_idx[min_deg_idx]
           chosen_global = cand_indices[chosen_local]
           added.append(chosen_global)
           available_sub[chosen_local] = False
           neigh_local = np.where(conflict_sub[chosen_local])[0]
           available_sub[neigh_local] = False
   