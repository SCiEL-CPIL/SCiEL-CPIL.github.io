seed_roots = 4.0 * np.arange(1, 13)
seed_score = np.exp(np.mean(np.log(1 + seed_roots)))
current_score = np.exp(np.mean(np.log(1 + roots)))
return current_score / seed_score
