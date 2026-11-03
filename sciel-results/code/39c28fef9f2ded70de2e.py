def construct():
    t0 = time.time()
    deadline = t0 + 140.0
    rng = np.random.default_rng(20240517)
    best_h = _rand_balanced(rng)
    best_val = _F(best_h)
    restart = 0
    while time.time() < deadline - 12:
        restart += 1
        if restart == 1 or rng.random() < 0.5:
            h = _rand_balanced(rng)
        else:
            h = best_h.copy()
            for _ in range(3):
                ones = np.nonzero(h>0.5)[0]; zeros=np.nonzero(h<0.5)[0]
                a=ones[rng.integers(len(ones))]; b=zeros[rng.integers(len(zeros))]
                h[a]=0.0; h[b]=1.0
        # SA
        h, v = _sa(h, rng, deadline)
        # local search
        h, v = _local_search(h, deadline, rng)
        if v < best_val:
            best_val = v
            best_h = h.copy()
    # final intensive local search on best
    best_h, best_val = _local_search(best_h, deadline, rng)
    return best_h.astype(float)
