<<<<<<< SEARCH
    # 25. Weighted: 0.5*downstream + 0.3*slack + 0.2*successor
    p_downstream2 = _normalize(downstream_time)
    p_slack = _normalize(slack_priority)
    p_succ3 = _normalize(succ_count.astype(float))
    cand_25 = 0.5 * p_downstream2 + 0.3 * p_slack + 0.2 * p_succ3
    candidates.append(cand_25)
    
    return candidates
=======
    # 25. Weighted: 0.5*downstream + 0.3*slack + 0.2*successor
    p_downstream2 = _normalize(downstream_time)
    p_slack = _normalize(slack_priority)
    p_succ3 = _normalize(succ_count.astype(float))
    cand_25 = 0.5 * p_downstream2 + 0.3 * p_slack + 0.2 * p_succ3
    candidates.append(cand_25)
    
    # 26. Pure downstream time (position weight) - classic heuristic
    candidates.append(downstream_time.copy())
    
    # 27. Slack-based with time emphasis: time * (1 + 2*normalized_slack_inv)
    slack_inv = _normalize(slack_priority)
    cand_27 = times * (1.0 + 2.0 * slack_inv)
    candidates.append(cand_27)
    
    # 28. Geometric mean of time and downstream time
    cand_28 = np.sqrt(times * downstream_time)
    candidates.append(cand_28)
    
    # 29. Harmonic mean style: 2*time*downstream / (time + downstream)
    denom_29 = times + downstream_time + 1e-9
    cand_29 = 2.0 * times * downstream_time / denom_29
    candidates.append(cand_29)
    
    # 30. Time * (1 + 0.3*normalized_bwd) - backward path influence
    p_bwd = _normalize(bwd)
    cand_30 = times * (1.0 + 0.3 * p_bwd)
    candidates.append(cand_30)
    
    # 31. Critical path with slack adjustment
    crit_slack = crit * (1.0 + 0.2 * _normalize(slack_priority))
    candidates.append(crit_slack)
    
    # 32. Downstream time squared (emphasize tasks with large downstream subtrees)
    candidates.append(downstream_time ** 2)
    
    # 33. Combined: 0.4*downstream + 0.3*crit + 0.2*time + 0.1*slack
    p_downstream3 = _normalize(downstream_time)
    p_crit2 = _normalize(crit)
    p_time3 = _normalize(times)
    p_slack2 = _normalize(slack_priority)
    cand_33 = 0.4 * p_downstream3 + 0.3 * p_crit2 + 0.2 * p_time3 + 0.1 * p_slack2
    candidates.append(cand_33)
    
    # 34. Bottleneck-focused: tasks whose time is close to cycle_time get boosted
    ct = instance["cycle_time"]
    bottleneck_proximity = times / max(ct, 1e-9)
    bottleneck_proximity = np.clip(bottleneck_proximity, 0, 1)
    cand_34 = downstream_time * (1.0 + 1.5 * bottleneck_proximity)
    candidates.append(cand_34)
    
    # 35. Successor-weighted critical path
    p_succ4 = _normalize(succ_count.astype(float))
    cand_35 = crit * (1.0 + 0.5 * p_succ4)
    candidates.append(cand_35)
    
    # 36. Inverse slack: 1/(1+slack) to emphasize low-slack tasks
    slack_safe = slack + 1.0
    cand_36 = 1.0 / slack_safe
    candidates.append(cand_36)
    
    # 37. Time * inverse slack
    cand_37 = times / slack_safe
    candidates.append(cand_37)
    
    # 38. Downstream time * (1 + 0.2 * time/cycle_time)
    time_ratio = times / max(ct, 1e-9)
    cand_38 = downstream_time * (1.0 + 0.2 * time_ratio)
    candidates.append(cand_38)
    
    # 39. Weighted sum with emphasis on both time and downstream
    cand_39 = 0.6 * _normalize(times) + 0.4 * _normalize(downstream_time)
    candidates.append(cand_39)
    
    # 40. Log-transformed downstream time (smooths large values)
    cand_40 = np.log1p(downstream_time)
    candidates.append(cand_40)
    
    return candidates
>>>>>>> REPLACE

