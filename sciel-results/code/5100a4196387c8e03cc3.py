import numpy as np
import time

N = 600

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s margin from 150s limit

    # Candidate 1: Uniform (Seed)
    h_uniform = np.ones(N) / N
    score_uniform = np.max(np.convolve(h_uniform, h_uniform, mode='full'))

    # Candidate 2: Gaussian
    # Try a few sigmas
    best_h = h_uniform
    best_score = score_uniform
    
    x = np.linspace(-0.25, 0.25, N)
    for sigma in [0.05, 0.1, 0.15, 0.2]:
        g = np.exp(-0.5 * (x / sigma) ** 2)
        g_sum = np.sum(g)
        if g_sum == 0: continue
        h_gauss = g / g_sum
        conv = np.convolve(h_gauss, h_gauss, mode='full')
        score_gauss = np.max(conv)
        if score_gauss < best_score:
            best_score = score_gauss
            best_h = h_gauss

    # Candidate 3: Semicircle
    # Wigner semicircle density: (2/pi) * sqrt(1 - (2x)^2) for |x| < 0.5
    # Here support is [-0.25, 0.25], so scale x by 4 to map to [-1, 1]
    # Actually, standard semicircle on [-1, 1] is (2/pi)sqrt(1-x^2).
    # Map our x to [-1, 1]: u = 4*x.
    u = 4 * x
    mask = np.abs(u) <= 1.0
    h_semi = np.zeros(N)
    h_semi[mask] = (2/np.pi) * np.sqrt(1 - u[mask]**2)
    h_semi_sum = np.sum(h_semi)
    if h_semi_sum > 0:
        h_semi = h_semi / h_semi_sum
        conv_semi = np.convolve(h_semi, h_semi, mode='full')
        score_semi = np.max(conv_semi)
        if score_semi < best_score:
            best_score = score_semi
            best_h = h_semi

    # Candidate 4: Cosine (Hann window)
    # h_i = sin^2(pi * i / (N-1)) or similar
    t = np.linspace(0, np.pi, N)
    h_cos = np.sin(t) ** 2
    h_cos_sum = np.sum(h_cos)
    if h_cos_sum > 0:
        h_cos = h_cos / h_cos_sum
        conv_cos = np.convolve(h_cos, h_cos, mode='full')
        score_cos = np.max(conv_cos)
        if score_cos < best_score:
            best_score = score_cos
            best_h = h_cos

    # Check time
    if time.time() > deadline:
        return list(best_h)

    # Try a simple local search on the Gaussian sigma if time permits
    # We already tried a few sigmas. Let's refine the best sigma found.
    # But the loop above is fast.
    
    return list(best_h)
