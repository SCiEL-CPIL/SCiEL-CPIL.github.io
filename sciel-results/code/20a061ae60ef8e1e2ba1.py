if delta > T * 50: # exp(-50) is tiny
    accept = False
else:
    accept = rng.random() < math.exp(-delta / T)
