import time
import sys
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin
    
    # Precompute prime factorization vectors for 2..128
    # Primes up to 128
    primes = []
    for p in range(2, 129):
        is_p = True
        for q in primes:
            if q * q > p:
                break
            if p % q == 0:
                is_p = False
                break
        if is_p:
            primes.append(p)
    
    prime_to_idx = {p: i for i, p in enumerate(primes)}
    num_primes = len(primes)
    
    # Precompute exponent vector for each number 2..128
    exp_vectors = [None] * 129  # index 0,1 unused
    for n in range(2, 129):
        vec = [0] * num_primes
        temp = n
        for i, p in enumerate(primes):
            while temp % p == 0:
                vec[i] += 1
                temp //= p
        # temp should be 1 since we have all primes up to 128
        exp_vectors[n] = tuple(vec)
    
    def is_valid_seq(seq):
        """Check if all consecutive block products are distinct using exponent vectors."""
        n = len(seq)
        if n == 0:
            return True
        seen = set()
        for i in range(n):
            vec = [0] * num_primes
            for j in range(i, n):
                # Add exp_vectors[seq[j]] to vec
                ej = exp_vectors[seq[j]]
                for k in range(num_primes):
                    vec[k] += ej[k]
                t = tuple(vec)
                if t in seen:
                    return False
                seen.add(t)
        return True
    
    # Start with all primes (guaranteed valid? Let's verify)
    # Actually, primes are not guaranteed valid together. Let's check.
    # But primes are a good starting point.
    prime_seq = sorted(primes)
    if not is_valid_seq(prime_seq):
        # Fallback: build greedily
        prime_seq = []
        for p in primes:
            cand = prime_seq + [p]
            if is_valid_seq(cand):
                prime_seq = cand
    
    best_seq = list(prime_seq)
    best_len = len(best_seq)
    
    # Beam search: try to insert composites
    composites = [x for x in range(2, 129) if x not in prime_to_idx]
    
    # Beam: list of sequences, each is a tuple
    beam = [tuple(best_seq)]
    beam_width = 50
    
    # Precompute which composites can be tried
    for step in range(200):  # Max 200 insertion steps
        if time.time() > deadline:
            break
        
        new_beam = []
        seen_seqs = set()
        
        for seq in beam:
            if time.time() > deadline:
                break
            seq_list = list(seq)
            seq_set = set(seq_list)
            
            # Try inserting each composite not in seq
            for x in composites:
                if x in seq_set:
                    continue
                # Try inserting x at each position
                # Since sequence must be strictly increasing, x can only go in positions
                # where prev < x < next
                # Find valid insertion positions
                pos = 0
                while pos <= len(seq_list):
                    if pos < len(seq_list) and seq_list[pos] < x:
                        pos += 1
                    else:
                        break
                # pos is the first index where seq_list[pos] >= x
                # So x can be inserted at position pos (if pos == 0 or seq_list[pos-1] < x)
                # Actually, since strictly increasing, x must satisfy:
                # (pos == 0 or seq_list[pos-1] < x) and (pos == len(seq_list) or seq_list[pos] > x)
                # But we found pos as first index with seq_list[pos] >= x, so seq_list[pos-1] < x is guaranteed
                # And we need seq_list[pos] > x, i.e., seq_list[pos] != x (which is true since x not in seq)
                # So only one valid position: pos
                
                new_seq_list = seq_list[:pos] + [x] + seq_list[pos:]
                new_seq = tuple(new_seq_list)
                
                if new_seq in seen_seqs:
                    continue
                seen_seqs.add(new_seq)
                
                if is_valid_seq(new_seq):
                    new_beam.append(new_seq)
            
            # Also try not inserting anything (keep current)
            if seq not in seen_seqs:
                seen_seqs.add(seq)
                new_beam.append(seq)
        
        if not new_beam:
            break
        
        # Sort by length descending, keep top beam_width
        new_beam.sort(key=lambda s: -len(s))
        beam = new_beam[:beam_width]
        
        # Update best
        if len(beam[0]) > best_len:
            best_len = len(beam[0])
            best_seq = list(beam[0])
        
        # If no improvement in last few steps, break
        if len(beam[0]) < best_len:
            break
    
    return best_seq
