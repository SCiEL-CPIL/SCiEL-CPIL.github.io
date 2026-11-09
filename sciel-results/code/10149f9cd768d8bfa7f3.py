<<<<<<< SEARCH
    # Weighted combination: blend static criticality with dynamic resource awareness
    scores = (0.30 * norm_float_inv +
              0.15 * norm_tail +
              0.10 * norm_duration +
              0.05 * norm_demand +
              0.04 * norm_ratio +
              0.02 * norm_succ +
              0.01 * norm_pred +
              0.33 * dynamic_scores)
    
    return scores
=======
    # Weighted combination: blend static criticality with dynamic resource awareness
    # Increase weight on dynamic scores since they capture resource bottlenecks directly
    scores = (0.25 * norm_float_inv +
              0.12 * norm_tail +
              0.08 * norm_duration +
              0.04 * norm_demand +
              0.03 * norm_ratio +
              0.02 * norm_succ +
              0.01 * norm_pred +
              0.45 * dynamic_scores)
    
    return scores
>>>>>>> REPLACE
