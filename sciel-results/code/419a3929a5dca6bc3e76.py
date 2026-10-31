import numpy as np
import time

N = 100
TIME_LIMIT = 140

def _compute_overlaps(h):
    n = len(h)
    ovs = np.empty(n - 1)
    for k in range(1, n):
        ovs[k - 1] = np.dot(h, 1.0 - np.roll(h, -k))
    return ovs

def _max_overlap(h):
    return float(np.max(_compute_overlaps(h)))

def _objective_and_grad(h, t=50.0):
    n = len(h)
    ovs = _compute_overlaps(h)
    m = np.max(ovs)
    exps = np.exp(t * (ovs - m))
    total = np.sum(exps)
    weights = exps / total
    val = m + np.log(total) / t

    grad = np.zeros(n)
    for k in range(1, n):
        w = weights[k - 1]
        if w > 1e-12:
            grad += w * ((1.0 - np.roll(h, -k)) - np.roll(h, k))

    return val, grad

def _project(h):
    h = np.clip(h, 0.0, 1.0)
    for _ in range(1000):
        s = np.sum(h)
        diff = N / 2.0 - s
        if abs(diff) < 1e-13:
            break
        h = h + diff / N
        h = np.clip(h, 0.0, 1.0)
    return h

def construct():
    start_time = time.time()
    rng = np.random.RandomState(42)

    best_h = np.full(N, 0.5)
    best_val = _max_overlap(best_h)

    for trial in range(15):
        if time.time() - start_time > TIME_LIMIT - 10:
            break

        if trial == 0:
            h = np.full(N, 0.5)
        elif trial == 1:
            h = np.array([1.0 if i % 2 == 0 else 0.0 for i in range(N)])
        elif trial == 2:
            h = np.array([1.0] * 50 + [0.0] * 50)
        elif trial == 3:
            h = np.array([1.0] * 25 + [0.0] * 25 + [1.0] * 25 + [0.0] * 25)
        elif trial == 4:
            h = np.array([1.0] * 10 + [0.0] * 10) * 5
        else:
            h = rng.uniform(0.0, 1.0, N)

        h = _project(h)

        t = 50.0
        lr = 0.005
        for step in range(400):
            if time.time() - start_time > TIME_LIMIT - 10:
                break

            val, grad = _objective_and_grad(h, t)
            h_new = h - lr * grad
            h_new = _project(h_new)

            val_new = _max_overlap(h_new)
            if val_new < val or step < 5:
                h = h_new

            if step % 25 == 0 and step > 0:
                t = max(15.0, t * 0.88)
                lr *= 0.96

        val = _max_overlap(h)
        if val < best_val:
            best_val = val
            best_h = h.copy()

    return _project(best_h)
