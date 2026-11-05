  for p in uncovered_points:
      temp = np.zeros(total_points, dtype=bool)
      temp[coverage_sets[p]] = True
      candidate_mask &= temp
  