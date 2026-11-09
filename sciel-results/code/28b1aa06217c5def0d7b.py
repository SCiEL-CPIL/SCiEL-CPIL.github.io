    beta = 0.2 * (1.0 + f_best / 10.0)
    lcb_scores = -mu + beta * sigma_safe
    ts_weight = 0.7
    scores = ts_weight * (-soft_min) + (1 - ts_weight) * lcb_scores
