"""Circle packing in perimeter-4 rectangle: pack 21 circles, maximize sum of radii."""
import numpy as np
import time
import random

N = 21

# EVOLVE-BLOCK-START

def construct():
    start_time = time.time()
    deadline = start_time + 145  # Safety margin before 150s limit

    def get_rect_dims(t):
        t = np.clip(t, 0.05, 0.95)
        w = 2.0 * t
        h = 2.0 * (1.0 - t)
        return w, h

    def make_initial(t_init=0.5, jitter=0.0):
        """Create initial state using grid pattern."""
        w, h = get_rect_dims(t_init)
        state = np.zeros(1 + 3 * N)
        state[0] = t_init

        # Try different grid arrangements: 5x5 (25 cells, use 21), 4x6, etc.
        # Use 5x5 grid
        side = 5
        idx = 0
        for i in range(side):
            for j in range(side):
                if idx >= N:
                    break
                gx = (i + 0.5) / side * w
                gy = (j + 0.5) / side * h
                r = 0.5 / side * min(w, h) * 0.92
                state[1 + 3 * idx] = gx
                state[1 + 3 * idx + 1] = gy
                state[1 + 3 * idx + 2] = r
                idx += 1

        if jitter > 0:
            for i in range(N):
                state[1 + 3 * i] += random.gauss(0, jitter * min(w, h))
                state[1 + 3 * i + 1] += random.gauss(0, jitter * min(w, h))
                state[1 + 3 * i + 2] *= random.uniform(0.85, 1.0)
            state[0] += random.gauss(0, 0.05)

        return state

    def evaluate(state):
        """Returns (score, penalty). Score = sum_r - penalty."""
        t = state[0]
        w, h = get_rect_dims(t)

        x = state[1:1 + 3 * N:3]
        y = state[2:1 + 3 * N:3]
        r = state[3:1 + 3 * N:3]

        sum_r = np.sum(r)
        penalty = 0.0

        # Boundary violations
        viol = np.maximum(0, r - x) + np.maximum(0, x + r - w) + \
               np.maximum(0, r - y) + np.maximum(0, y + r - h)
        penalty += 500.0 * np.sum(viol)

        # Overlap violations (smooth-ish)
        for i in range(N):
            for j in range(i + 1, N):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                dist = np.sqrt(dx * dx + dy * dy)
                min_dist = r[i] + r[j]
                if dist < min_dist:
                    penalty += 1000.0 * (min_dist - dist)

        return sum_r - penalty, penalty

    def check_valid(state, eps=1e-7):
        t = state[0]
        w, h = get_rect_dims(t)
        x = state[1:1 + 3 * N:3]
        y = state[2:1 + 3 * N:3]
        r = state[3:1 + 3 * N:3]

        if np.any(r < 1e-6):
            return False
        if np.any(x - r < -eps) or np.any(x + r > w + eps) or \
           np.any(y - r < -eps) or np.any(y + r > h + eps):
            return False
        for i in range(N):
            for j in range(i + 1, N):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                if dx * dx + dy * dy < (r[i] + r[j] - eps) ** 2:
                    return False
        return True

    def gradient_descent(state, time_limit, lr_init=0.01):
        """Gradient descent with adaptive learning rate."""
        best_state = state.copy()
        best_score, _ = evaluate(best_state)
        lr = lr_init
        no_improve = 0
        max_no_improve = 5000

        while time.time() < time_limit:
            if time.time() > time_limit - 1:
                break

            t = best_state[0]
            w, h = get_rect_dims(t)
            x = best_state[1:1 + 3 * N:3].copy()
            y = best_state[2:1 + 3 * N:3].copy()
            r = best_state[3:1 + 3 * N:3].copy()

            # Compute gradient numerically (central differences for key vars)
            # Use finite differences for the full state
            grad = np.zeros_like(best_state)
            h_fd = 1e-5

            # For efficiency, only compute gradient for a subset of variables per step
            # Actually, let's do full gradient but with forward differences for speed
            for idx in range(len(best_state)):
                orig = best_state[idx]
                best_state[idx] = orig + h_fd
                sp = evaluate(best_state)[0]
                best_state[idx] = orig - h_fd
                sm = evaluate(best_state)[0]
                best_state[idx] = orig
                grad[idx] = (sp - sm) / (2 * h_fd)

            # Update
            new_state = best_state - lr * grad
            # Clamp t
            new_state[0] = np.clip(new_state[0], 0.05, 0.95)
            # Clamp radii positive
            for i in range(N):
                ridx = 3 + 3 * i
                new_state[ridx] = max(new_state[ridx], 1e-5)
                w, h = get_rect_dims(new_state[0])
                new_state[ridx] = min(new_state[ridx], 0.5 * min(w, h))

            new_score, _ = evaluate(new_state)

            if new_score > best_score:
                best_state = new_state
                best_score = new_score
                no_improve = 0
                lr = min(lr * 1.05, 0.05)
            else:
                no_improve += 1
                lr *= 0.95

            if no_improve > max_no_improve:
                break

        return best_state, best_score

    def simulated_annealing(state, time_limit):
        """Simulated annealing."""
        best_state = state.copy()
        best_score, _ = evaluate(best_state)
        T = 0.3
        T_end = 1e-6
        alpha = 0.9997

        while time.time() < time_limit:
            if time.time() > time_limit - 1:
                break

            new_state = best_state.copy()
            var_idx = random.randint(0, len(new_state) - 1)

            if var_idx == 0:
                new_state[0] += random.gauss(0, 0.02)
                new_state[0] = np.clip(new_state[0], 0.05, 0.95)
            else:
                circle_idx = (var_idx - 1) // 3
                comp = (var_idx - 1) % 3
                scale = T / 0.3 * 0.08 + 0.005
                if comp == 0:
                    new_state[var_idx] += random.gauss(0, scale)
                elif comp == 1:
                    new_state[var_idx] += random.gauss(0, scale)
                else:
                    new_state[var_idx] += random.gauss(0, scale * 0.3)
                    w, h = get_rect_dims(new_state[0])
                    new_state[var_idx] = np.clip(new_state[var_idx], 1e-5, 0.5 * min(w, h))

            new_score, _ = evaluate(new_state)

            if new_score > best_score or random.random() < np.exp((new_score - best_score) / T):
                best_state = new_state
                best_score = new_score

            T *= alpha
            if T < T_end:
                T = T_end

        return best_state, best_score

    def polish(state, time_limit):
        """High-precision polish: alternate small gradient steps and SA."""
        best_state = state.copy()
        best_score, _ = evaluate(best_state)

        while time.time() < time_limit:
            if time.time() > time_limit - 2:
                break

            # Small gradient step
            t = best_state[0]
            w, h = get_rect_dims(t)
            x = best_state[1:1 + 3 * N:3].copy()
            y = best_state[2:1 + 3 * N:3].copy()
            r = best_state[3:1 + 3 * N:3].copy()

            # Compute gradient for positions and radii (not t)
            h_fd = 1e-6
            for i in range(N):
                for comp in range(3):
                    idx = 1 + 3 * i + comp
                    orig = best_state[idx]
                    best_state[idx] = orig + h_fd
                    sp = evaluate(best_state)[0]
                    best_state[idx] = orig - h_fd
                    sm = evaluate(best_state)[0]
                    best_state[idx] = orig
                    g = (sp - sm) / (2 * h_fd)
                    best_state[idx] -= 0.005 * g

            best_state[0] = np.clip(best_state[0], 0.05, 0.95)
            for i in range(N):
                ridx = 3 + 3 * i
                w, h = get_rect_dims(best_state[0])
                best_state[ridx] = np.clip(best_state[ridx], 1e-5, 0.5 * min(w, h))

            new_score, _ = evaluate(best_state)
            if new_score > best_score:
                best_score = new_score
            else:
                # Revert
                best_state[1:1 + 3 * N:3] = x
                best_state[2:1 + 3 * N:3] = y
                best_state[3:1 + 3 * N:3] = r

        return best_state, best_score

    # Main algorithm
    best_state = None
    best_score = -np.inf

    # Phase 1: Multiple SA restarts with different initial configs
    configs = [
        (0.5, 0.0),   # square, no jitter
        (0.6, 0.05),  # wider
        (0.4, 0.05),  # taller
        (0.55, 0.1),  # slightly wider with jitter
        (0.45, 0.1),  # slightly taller with jitter
    ]

    # Allocate time: ~70% SA, ~20% GD, ~10% polish
    total_time = 145
    sa_time_per = (total_time * 0.6) / len(configs)

    for ci, (t_init, jitter) in enumerate(configs):
        if time.time() > deadline - 20:
            break
        state = make_initial(t_init, jitter)
        sa_deadline = min(time.time() + sa_time_per, deadline - 25)
        state, score = simulated_annealing(state, sa_deadline)
        if score > best_score:
            best_score = score
            best_state = state.copy()

    # Phase 2: Gradient descent from best
    if time.time() < deadline - 15:
        gd_deadline = min(time.time() + 25, deadline - 15)
        state, score = gradient_descent(best_state.copy(), gd_deadline, lr_init=0.005)
        if score > best_score:
            best_score = score
            best_state = state.copy()

    # Phase 3: More SA from improved state
    if time.time() < deadline - 10:
        sa_deadline = min(time.time() + 15, deadline - 10)
        state, score = simulated_annealing(best_state.copy(), sa_deadline)
        if score > best_score:
            best_score = score
            best_state = state.copy()

    # Phase 4: Polish
    if time.time() < deadline - 5:
        polish_deadline = min(time.time() + 10, deadline - 5)
        state, score = polish(best_state.copy(), polish_deadline)
        if score > best_score and check_valid(state):
            best_score = score
            best_state = state.copy()

    # Final validity check and repair
    if best_state is None:
        best_state = make_initial(0.5, 0.0)
    elif not check_valid(best_state):
        repair = best_state.copy()
        for _ in range(100):
            if check_valid(repair):
                break
            for i in range(N):
                repair[3 + 3 * i] *= 0.97
        if check_valid(repair):
            best_state = repair
        else:
            best_state = make_initial(0.5, 0.0)

    # Extract result
    result = np.zeros((N, 3))
    for i in range(N):
        result[i, 0] = best_state[1 + 3 * i]
        result[i, 1] = best_state[1 + 3 * i + 1]
        result[i, 2] = best_state[1 + 3 * i + 2]

    return result

# EVOLVE-BLOCK-END
