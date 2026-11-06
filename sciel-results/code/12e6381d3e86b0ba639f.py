    # 19. Weighted: times + 2*cp + 0.5*trans_succ
    candidates.append(times + 2*cp + 0.5*trans_succ)
    
    # 20. cp + sum_succ_times
    candidates.append(cp + sum_succ_times)
    # 21. cp * (1 + sum_succ_times)
    candidates.append(cp * (1 + sum_succ_times))
    # 22. times + sum_succ_times
    candidates.append(times + sum_succ_times)
    # 23. cp + 2*sum_succ_times
    candidates.append(cp + 2*sum_succ_times)

    # Evaluate each candidate and find the best
