import numpy as np
import time

N = 400

def construct():
    """
    Returns a sequence of signed step heights (length N) that minimizes
    the upper bound on C3 (interpreted as peak autocorrelation sidelobe).
    """
    start_time = time.time()
    deadline = start_time + 130  # Safety margin (150s limit)

    def compute_peak_sidelobe(h):
        """
        Computes the maximum absolute value of the autocorrelation
        at non-zero lags using linear autocorrelation.
        """
        n = len(h)
        nfft = 1 << (2 * n - 1).bit_length()
        H = np.fft.rfft(h, n=nfft)
        R = np.fft.irfft(H * np.conj(H), n=nfft)
        # Linear autocorrelation: R[0..n-1] are positive lags, R[nfft-n+1..nfft-1] negative lags
        sidelobes_pos = R[1:n]
        sidelobes_neg = R[nfft - n + 1:nfft]
        all_sidelobes = np.concatenate([sidelobes_pos, sidelobes_neg])
        return np.max(np.abs(all_sidelobes))

    def thue_morse_sequence(n):
        """Generate Thue-Morse sequence of length n, mapped to ±1."""
        seq = np.zeros(n, dtype=np.int8)
        for i in range(n):
            x = i
            cnt = 0
            while x:
                cnt += x & 1
                x >>= 1
            seq[i] = 1 if cnt % 2 == 0 else -1
        return seq

    def hadamard_row(n, row_idx):
        """Generate a row of the Hadamard matrix (Walsh-Hadamard)."""
        seq = np.zeros(n, dtype=np.int8)
        for i in range(n):
            # XOR-based Hadamard
            val = 1
            x = i ^ row_idx
            while x:
                val *= -1
                x &= x - 1
            seq[i] = val
        return seq

    def block_flip(h, start, length):
        """Flip a contiguous block of h[start:start+length]."""
        h[start:start+length] *= -1

    def block_move(h, src_start, src_len, dst_start):
        """Move a block from src to dst position."""
        if dst_start >= N:
            return
        block = h[src_start:src_start+src_len].copy()
        # Remove from source
        h[src_start:src_start+src_len] = h[N-src_len:N]
        h[N-src_len:N] = 0  # placeholder
        # Shift to make room
        if dst_start < src_start:
            h[dst_start+src_len:src_start] = h[dst_start:src_start-src_len]
            h[dst_start:dst_start+src_len] = block
        elif dst_start > src_start:
            h[dst_start:dst_start+src_len] = h[dst_start-src_len:dst_start]
            h[dst_start-src_len:dst_start] = block
        else:
            h[src_start:src_start+src_len] = block
        # Fix the tail
        h[N-src_len:N] = h[0:src_len].copy() if src_start == 0 else h[src_start:src_start+src_len].copy()

    def find_blocks(h):
        """Find contiguous blocks of same sign in h."""
        blocks = []
        i = 0
        while i < N:
            j = i
            while j < N and h[j] == h[i]:
                j += 1
            blocks.append((i, j - i, h[i]))
            i = j
        return blocks

    def block_based_search(h, max_iterations=5000):
        """
        Block-based local search:
        1. Flip a random block
        2. Merge/split adjacent blocks
        3. Move a block to a new position
        """
        current_score = compute_peak_sidelobe(h)
        improved = True
        iterations = 0

        while improved and iterations < max_iterations:
            improved = False
            iterations += 1

            if time.time() > deadline - 5:
                break

            # Try flipping random blocks of various sizes
            for _ in range(15):
                if time.time() > deadline - 5:
                    break
                # Choose block size: 1, 2, 4, 8, 16, or random up to N//4
                r = np.random.random()
                if r < 0.2:
                    blen = 1
                elif r < 0.4:
                    blen = 2
                elif r < 0.6:
                    blen = 4
                elif r < 0.8:
                    blen = 8
                else:
                    blen = np.random.randint(1, N // 4 + 1)
                
                blen = min(blen, N)
                start = np.random.randint(0, N - blen + 1)
                
                block_flip(h, start, blen)
                new_score = compute_peak_sidelobe(h)
                if new_score < current_score:
                    current_score = new_score
                    improved = True
                else:
                    block_flip(h, start, blen)  # revert

            # Try merging adjacent same-sign blocks (no-op if already merged)
            # or splitting a block
            for _ in range(5):
                if time.time() > deadline - 5:
                    break
                # Pick a random position and try to extend or contract a block
                pos = np.random.randint(N)
                # Find the block containing pos
                val = h[pos]
                # Extend left
                left = pos
                while left > 0 and h[left-1] == val:
                    left -= 1
                # Extend right
                right = pos
                while right < N - 1 and h[right+1] == val:
                    right += 1
                
                # Try flipping just the leftmost element of this block
                if left < right:
                    h[left] *= -1
                    new_score = compute_peak_sidelobe(h)
                    if new_score < current_score:
                        current_score = new_score
                        improved = True
                    else:
                        h[left] *= -1
                
                # Try flipping just the rightmost element
                if left < right:
                    h[right] *= -1
                    new_score = compute_peak_sidelobe(h)
                    if new_score < current_score:
                        current_score = new_score
                        improved = True
                    else:
                        h[right] *= -1

        return h, current_score

    def simulated_annealing_blocks(h, iterations=3000, T_start=2.0, T_end=0.05):
        """
        Simulated annealing with block moves.
        """
        current_score = compute_peak_sidelobe(h)
        T = T_start

        for iteration in range(iterations):
            if time.time() > deadline - 10:
                break

            # Choose move type
            r = np.random.random()
            if r < 0.5:
                # Flip a block
                blen = np.random.choice([1, 2, 4, 8, 16, 32, 64])
                blen = min(blen, N)
                start = np.random.randint(0, N - blen + 1)
                block_flip(h, start, blen)
                new_score = compute_peak_sidelobe(h)
                delta = new_score - current_score
                if delta < 0 or np.random.random() < np.exp(-delta / T):
                    current_score = new_score
                else:
                    block_flip(h, start, blen)  # revert

            elif r < 0.8:
                # Swap two blocks
                blen1 = np.random.randint(1, 17)
                blen2 = np.random.randint(1, 17)
                start1 = np.random.randint(0, N - blen1 + 1)
                start2 = np.random.randint(0, N - blen2 + 1)
                if abs(start1 - start2) < max(blen1, blen2):
                    continue  # overlapping, skip
                block1 = h[start1:start1+blen1].copy()
                block2 = h[start2:start2+blen2].copy()
                h[start1:start1+blen1] = block2[:blen1] if blen2 >= blen1 else block2
                h[start2:start2+blen2] = block1[:blen2] if blen1 >= blen2 else block1
                # Handle size mismatch
                if blen1 != blen2:
                    if blen1 > blen2:
                        h[start2+blen2:start2+blen1] = block2[blen2:] if blen2 > 0 else block2
                    else:
                        h[start1+blen1:start1+blen2] = block1[blen1:] if blen1 > 0 else block1
                new_score = compute_peak_sidelobe(h)
                delta = new_score - current_score
                if delta < 0 or np.random.random() < np.exp(-delta / T):
                    current_score = new_score
                else:
                    # Revert
                    h[start1:start1+blen1] = block1
                    h[start2:start2+blen2] = block2

            else:
                # Reverse a block
                blen = np.random.randint(2, 33)
                blen = min(blen, N)
                start = np.random.randint(0, N - blen + 1)
                h[start:start+blen] = h[start:start+blen][::-1]
                new_score = compute_peak_sidelobe(h)
                delta = new_score - current_score
                if delta < 0 or np.random.random() < np.exp(-delta / T):
                    current_score = new_score
                else:
                    h[start:start+blen] = h[start:start+blen][::-1]  # revert

            # Cooling
            T = T_start * (T_end / T_start) ** (iteration / iterations)

        return h, current_score

    best_h = None
    best_score = float('inf')

    # Generate diverse initial sequences using structured patterns
    initial_sequences = []
    
    # Thue-Morse
    initial_sequences.append(thue_morse_sequence(N))
    
    # Hadamard rows (different rows give different patterns)
    for row in [1, 3, 7, 15, 31, 63, 127]:
        if row < N:
            initial_sequences.append(hadamard_row(N, row))
    
    # Random with structure
    np.random.seed(42)
    for _ in range(10):
        initial_sequences.append(np.random.choice([-1, 1], N))
    
    # Block-structured: alternating blocks of random lengths
    for _ in range(5):
        h = np.zeros(N, dtype=np.int8)
        pos = 0
        sign = np.random.choice([-1, 1])
        while pos < N:
            blen = np.random.randint(1, 21)
            blen = min(blen, N - pos)
            h[pos:pos+blen] = sign
            sign *= -1
            pos += blen
        initial_sequences.append(h)

    # Main search loop
    for init_idx, init_h in enumerate(initial_sequences):
        if time.time() > deadline - 20:
            break

        h = init_h.copy().astype(np.int8)

        # Phase 1: Simulated annealing with block moves
        h, sa_score = simulated_annealing_blocks(h, iterations=2000, T_start=2.0, T_end=0.05)

        # Phase 2: Block-based local search
        h, local_score = block_based_search(h, max_iterations=3000)

        # Update best
        final_score = compute_peak_sidelobe(h)
        if final_score < best_score:
            best_score = final_score
            best_h = h.copy()

        # Early termination if we've found a good solution
        if best_score < 0.5:
            # Do more refinement on the best
            if time.time() < deadline - 30:
                h = best_h.copy()
                h, _ = simulated_annealing_blocks(h, iterations=1000, T_start=0.5, T_end=0.01)
                h, _ = block_based_search(h, max_iterations=2000)
                final_score = compute_peak_sidelobe(h)
                if final_score < best_score:
                    best_score = final_score
                    best_h = h.copy()
            break

    if best_h is None:
        best_h = np.random.choice([-1, 1], N).astype(np.int8)

    return best_h
