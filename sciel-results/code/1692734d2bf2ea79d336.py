<<<<<<< SEARCH
    # 7. Highly composite
    cand7 = sorted([1, 2, 3, 4, 5, 6, 8, 9, 10, 12, 15, 16, 18, 20, 24, 25, 27, 30, 32, 36])
    candidates.append(cand7)
    
    # 8. Log-spaced
    cand8 = []
    for target_log in [i * 0.3 for i in range(20)]:
        target = int(round(2 ** target_log))
        best_smooth = min(smooth_numbers, key=lambda x: abs(x - target))
        if best_smooth not in cand8:
            cand8.append(best_smooth)
    cand8 = sorted(cand8[:20])
    candidates.append(cand8)
=======
    # 7. Highly composite
    cand7 = sorted([1, 2, 3, 4, 5, 6, 8, 9, 10, 12, 15, 16, 18, 20, 24, 25, 27, 30, 32, 36])
    candidates.append(cand7)
    
    # 8. Log-spaced
    cand8 = []
    for target_log in [i * 0.3 for i in range(20)]:
        target = int(round(2 ** target_log))
        best_smooth = min(smooth_numbers, key=lambda x: abs(x - target))
        if best_smooth not in cand8:
            cand8.append(best_smooth)
    cand8 = sorted(cand8[:20])
    candidates.append(cand8)
    
    # 9. Prime powers
    cand9 = []
    for p in [2, 3, 5, 7]:
        val = 1
        while val <= 10000 and len(cand9) < 20:
            cand9.append(val)
            val *= p
    cand9 = sorted(set(cand9)[:20])
    candidates.append(cand9)
    
    # 10. Mixed small and large smooth
    cand10 = sorted(rng.sample(smooth_numbers[:100], 10) + rng.sample(smooth_numbers[100:300], 10))
    if len(set(cand10)) == 20:
        candidates.append(cand10)
>>>>>>> REPLACE

