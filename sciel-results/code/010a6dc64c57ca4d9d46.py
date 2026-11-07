import numpy as np

def heuristics(prizes, weights):
    total_prize = prizes.sum(axis=0)
    eps = 1e-9
    efficiencies = total_prize[None, :] / (weights + eps)
    log_eff = np.log(efficiencies)
    geo_mean_eff = np.exp(log_eff.mean(axis=0))
    return geo_mean_eff
