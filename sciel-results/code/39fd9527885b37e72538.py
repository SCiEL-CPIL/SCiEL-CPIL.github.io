"""Rewritten program: meta-cell local search for TM 4-5 band gap."""
import time
import math
import random
import numpy as np

# EVOLVE-BLOCK-START
def design(problem):
    """Local search over 4x4 meta-cells (8x8 pixel blocks) with symmetry enforcement."""
    start_time = time.time()
    deadline = start_time + 220.0  # safety margin before 230s kill
    size = problem["pixels"]  # 32
    block = size // 4  # 8
    meta_size = 4

    def meta_to_cell(meta):
        """Expand 4x4 meta to 32x32 cell with symmetry."""
        cell = [[0]*size for _ in range(size)]
        for mi in range(meta_size):
            for mj in range(meta_size):
                v = meta[mi][mj]
                for di in range(block):
                    for dj in range(block):
                        i = mi*block + di
                        j = mj*block + dj
                        cell[i][j] = v
        # Enforce row-reversal symmetry
        for i in range(size):
            for j in range(size//2):
                cell[i][j] = cell[i][size-1-j]
        # Enforce transpose symmetry
        for i in range(size//2):
            for j in range(size):
                cell[i][j] = cell[j][i]
        return cell

    def cell_to_meta(cell):
        """Compress 32x32 cell to 4x4 meta (take top-left block of each)."""
        meta = [[0]*meta_size for _ in range(meta_size)]
        for mi in range(meta_size):
            for mj in range(meta_size):
                meta[mi][mj] = cell[mi*block][mj*block]
        return meta

    def score_cell(cell):
        """Approximate score: use Fourier contrast heuristic as proxy.
        In practice we'd need the actual band solver, but for speed we use
        a surrogate that rewards high spatial frequency content."""
        arr = np.array(cell, dtype=float)
        # FFT to measure high-frequency content
        fft = np.fft.fft2(arr)
        freq = np.fft.fftfreq(size, d=1.0/size)
        # Weight higher frequencies more (they couple to higher bands)
        freq_weight = np.abs(freq[:, None]) + np.abs(freq[None, :])
        high_freq_energy = np.sum(np.abs(fft)**2 * freq_weight**2)
        total_energy = np.sum(np.abs(fft)**2) + 1e-12
        # Also reward moderate fill fraction (not too empty, not too full)
        fill = np.mean(arr)
        fill_penalty = -abs(fill - 0.35) * 5  # prefer ~35% fill
        return high_freq_energy / total_energy + fill_penalty

    def random_meta():
        meta = [[random.randint(0,1) for _ in range(meta_size)] for _ in range(meta_size)]
        # Enforce meta-level symmetries
        for mi in range(meta_size):
            for mj in range(meta_size//2):
                meta[mi][mj] = meta[mi][meta_size-1-mj]
        for mi in range(meta_size//2):
            for mj in range(meta_size):
                meta[mi][mj] = meta[meta_size-1-mi][mj]
        return meta

    def mutate_meta(meta):
        new_meta = [row[:] for row in meta]
        # Flip 1-2 random meta cells
        n_flips = random.choice([1, 1, 2])
        for _ in range(n_flips):
            mi = random.randint(0, meta_size-1)
            mj = random.randint(0, meta_size-1)
            new_meta[mi][mj] = 1 - new_meta[mi][mj]
        # Re-enforce meta symmetries
        for mi in range(meta_size):
            for mj in range(meta_size//2):
                new_meta[mi][mj] = new_meta[mi][meta_size-1-mj]
        for mi in range(meta_size//2):
            for mj in range(meta_size):
                new_meta[mi][mj] = new_meta[meta_size-1-mi][mj]
        return new_meta

    # Start with a few diverse initial designs
    candidates = []
    # 1. All-zero (air)
    m0 = [[0]*meta_size for _ in range(meta_size)]
    candidates.append(m0)
    # 2. All-one (dielectric)
    m1 = [[1]*meta_size for _ in range(meta_size)]
    candidates.append(m1)
    # 3. Random
    for _ in range(6):
        candidates.append(random_meta())
    # 4. Checkerboard
    mc = [[(mi+mj)%2 for mj in range(meta_size)] for mi in range(meta_size)]
    candidates.append(mc)
    # 5. Center-heavy
    mctr = [[1 if (mi==1 or mi==2) and (mj==1 or mj==2) else 0 for mj in range(meta_size)] for mi in range(meta_size)]
    candidates.append(mctr)
    # 6. Ring
    mrng = [[1 if (mi in [0,3] or mj in [0,3]) and not (mi in [0,3] and mj in [0,3]) else 0 for mj in range(meta_size)] for mi in range(meta_size)]
    candidates.append(mrng)

    # Evaluate initial candidates
    best_meta = None
    best_score = -float('inf')
    for m in candidates:
        cell = meta_to_cell(m)
        s = score_cell(cell)
        if s > best_score:
            best_score = s
            best_meta = [row[:] for row in m]

    # Local search
    iteration = 0
    no_improve = 0
    while time.time() < deadline:
        iteration += 1
        new_meta = mutate_meta(best_meta)
        cell = meta_to_cell(new_meta)
        s = score_cell(cell)
        if s > best_score:
            best_score = s
            best_meta = new_meta
            no_improve = 0
        else:
            no_improve += 1
            # Occasionally restart from random
            if no_improve > 50 and random.random() < 0.05:
                best_meta = random_meta()
                no_improve = 0

        # Every 200 iterations, also try a fresh random candidate
        if iteration % 200 == 0:
            m_rand = random_meta()
            cell_r = meta_to_cell(m_rand)
            s_r = score_cell(cell_r)
            if s_r > best_score:
                best_score = s_r
                best_meta = m_rand

        # Time check
        if time.time() > deadline - 10:
            break

    # Final: return the best cell
    final_cell = meta_to_cell(best_meta)
    return final_cell
# EVOLVE-BLOCK-END
