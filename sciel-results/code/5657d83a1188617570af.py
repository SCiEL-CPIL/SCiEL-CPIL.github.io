  def repair_independent_set(in_set):
      valid_set = np.zeros(N4, dtype=bool)
      # Sort indices by degree descending? Or just iterate?
      # To maximize size, we might want to keep high-degree nodes? 
      # Actually, standard greedy repair:
      # Iterate through candidates. If candidate doesn't conflict with current valid set, add it.
      # Order matters. 
      # Let's sort candidates by degree descending to potentially keep 'harder' nodes? 
      # Or ascending? 
      # Usually, for independent set, removing low-degree nodes first is better for size?
      # No, if we are *selecting* from a candidate set, we want to select nodes that block few others.
      # So sort by degree ascending.
      candidates = np.where(in_set)[0]
      if len(candidates) == 0:
          return valid_set, 0
      
      # Sort by degree ascending (less restrictive first)
      candidates_sorted = candidates[np.argsort(neighbor_counts[candidates])]
      
      for v in candidates_sorted:
          # Check if v conflicts with any in valid_set
          # Use bitmask
          v_mask = neighbor_bitmasks[v]
          # Get indices of valid_set
          current_indices = np.where(valid_set)[0]
          if len(current_indices) == 0:
              valid_set[v] = True
              continue
          
          # Check conflicts
          conflict_found = False
          for u in current_indices:
              if (v_mask >> u) & 1:
                  conflict_found = True
                  break
          
          if not conflict_found:
              valid_set[v] = True
              
      return valid_set, int(valid_set.sum())
  