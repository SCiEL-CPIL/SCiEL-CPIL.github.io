import time
import random
import numpy as np

N = 18
DEADLINE = 85.0  # Safety margin before 90s kill

def _odd_even_transposition():
    """Generate an odd-even transposition sorting network for N wires."""
    comps = []
    for k in range(N):
        if k % 2 == 0:
            for i in range(0, N - 1, 2):
                comps.append((i, i + 1))
        else:
            for i in range(1, N - 1, 2):
                comps.append((i, i + 1))
    return comps

def _verify_network_bit_parallel(comps):
    """
    Verify if a sorting network sorts all 2^N Boolean inputs.
    Uses bit-parallelism: each of the 2^N inputs is represented as a bit in an integer.
    For each wire, we maintain an integer where bit k is 1 if wire is 1 for input k.
    """
    # For each wire i, state[i] is an integer where bit k represents the value of wire i 
    # for the k-th input (k from 0 to 2^N - 1).
    # Input k has bits based on k.
    # Specifically, for input k (0 to 2^N-1), the value at wire i is (k >> i) & 1.
    # So state[i] should have bit k set if (k >> i) & 1 == 1.
    
    # Let's construct state[i] efficiently.
    # state[i] is a bit vector of length 2^N.
    # We can generate these using numpy or pure Python integers.
    # 2^18 is 262144 bits, which is ~32KB per integer. 18 integers is ~576KB. Very small.
    
    # Generate state[i]
    # state[i] has 1s at positions k where (k & (1 << i)) != 0.
    # Pattern for bit i: 0...0 1...1 0...0 1...1 ...
    # Block size is 2^(i+1). First 2^i are 0, next 2^i are 1.
    
    states = [0] * N
    
    # Precompute masks? Or just build them.
    # Building 18 integers of 2^18 bits is fast.
    for i in range(N):
        # We need an integer with 1s at bits k where k has bit i set.
        # This is a periodic pattern.
        # Instead of building bit by bit, we can use a trick.
        # But 262144 iterations is fine in Python if we do it once.
        # Or we can use numpy to create a uint64 array and pack?
        # Pure Python integer construction:
        val = 0
        # The pattern repeats every 2^(i+1) bits.
        # In each block of 2^(i+1), the top 2^i bits are 1.
        # We can build this by shifting and ORing.
        # Let block_size = 1 << (i + 1)
        # Let half_block = 1 << i
        # The mask for one block is (half_block - 1) << i ? No.
        # Bits 0 to 2^i-1 are 0. Bits 2^i to 2^(i+1)-1 are 1.
        # So the mask for one block is ((1 << i) - 1) << i? No, that's lower bits.
        # We want bits i to 2i-1? No.
        # Let's just use a simple loop or numpy.
        
        # Using numpy to generate the bit vector for all inputs at once is faster for initialization.
        # But we are doing this once.
        
        # Let's try a pure Python approach with a precomputed pattern for each i.
        # Actually, we can just initialize states using the definition of the input.
        # Input k is the binary representation of k.
        # So for wire i, the value is the i-th bit of k.
        
        # Let's build the integer.
        # We can iterate k from 0 to 2^N - 1 and set bits? No, too slow (262k * 18).
        
        # Better: Use the structure.
        # For i=0: 010101...
        # For i=1: 00110011...
        # For i=2: 00001111...
        
        # We can construct this by repeating a pattern.
        # Pattern for i is: 0...0 (2^i times) 1...1 (2^i times).
        # Let P_i be the integer representing one period of length 2^(i+1).
        # P_i = sum(1 << j for j in range(i, 2*i)) ? No, range(i, 2*i) is i to 2i-1.
        # Wait, the period length is 2^(i+1).
        # The ones are at positions i, i+1, ..., 2*i - 1?
        # No. The bit position in the *integer* corresponds to the input index k.
        # So bit k of the integer is 1 if (k & (1<<i)) != 0.
        
        # Let's just generate the integers using a fast method.
        # We can use a recursive doubling or just a loop if it's fast enough.
        # 2^18 is 262144.
        # Creating 18 such integers might take ~10-50ms in Python if done naively.
        # Let's try to be smart.
        
        # We can use numpy to create a boolean array of shape (2^N,) for each i,
        # then convert to a big integer?
        # Or just use the fact that we will simulate the network.
        
        # Alternative: Don't precompute states. Just simulate the network on the "symbolic" inputs.
        # But we need to check if the output is sorted for ALL inputs.
        # The output for wire j is a bit vector O_j.
        # The network is valid if for all k, O_0[k] <= O_1[k] <= ... <= O_{N-1}[k].
        # This is equivalent to: (O_j & ~O_{j+1}) == 0 for all j.
        # i.e., O_j & ~O_{j+1} must be 0 for all j.
        
        # So we need to compute the output bit vectors.
        # How to initialize the input bit vectors efficiently?
        
        # Let's use a helper to generate the bit vector for wire i.
        # We can build it by repeating a block.
        # Block length L = 1 << (i+1)
        # Block pattern: lower 2^i bits are 0, upper 2^i bits are 1.
        # Block value B_i = ((1 << i) - 1) << i ? No.
        # Bits i..2i-1 are 1?
        # Let's check i=0. L=2. Bits 0..0 are 0, bit 1 is 1.
        # B_0 = 1 << 0 = 1? No, bit 1 is set. So B_0 = 2 (binary 10).
        # Wait, bit index 0 is LSB.
        # For i=0, k=0 (bit 0 of k is 0) -> 0. k=1 (bit 0 of k is 1) -> 1.
        # So the integer should have bit 1 set? No, the integer's bit k represents input k.
        # So for i=0, the integer has bits 1, 3, 5... set.
        # Integer = 0b101010...
        
        # For i=1, k=0(00)->0, k=1(01)->0, k=2(10)->1, k=3(11)->1.
        # Integer has bits 2, 3, 6, 7... set.
        # Integer = 0b11001100...
        
        # So for wire i, the integer is formed by repeating the pattern:
        # 0...0 (2^i zeros) 1...1 (2^i ones)
        # This pattern has length 2^(i+1).
        # We need to repeat this pattern 2^(N-i-1) times to fill 2^N bits.
        
        # We can build this by:
        # base = 0
        # for j in range(i, 2*i): base |= (1 << j)  # This creates the ones in the block?
        # No, the positions in the block are 0 to 2^(i+1)-1.
        # The ones are at positions i to 2*i - 1.
        # So base_block = sum(1 << p for p in range(i, 2*i))
        
        # Then we repeat this block.
        # total_bits = 2^N
        # repeats = total_bits // (2^(i+1))
        
        # We can construct the full integer by shifting and ORing.
        
        val = 0
        block_len = 1 << (i + 1)
        # Construct one block
        block_val = 0
        for p in range(i, 2 * i):
            block_val |= (1 << p)
        
        # Repeat block_val
        # We can do this efficiently.
        # val = 0
        # shift = 0
        # for r in range(repeats):
        #     val |= (block_val << shift)
        #     shift += block_len
        
        # This loop runs 2^(N-i-1) times.
        # For i=0, 2^17 = 131072 iterations. Too slow in Python.
        
        # We need a faster way to generate these integers.
        # We can use numpy to create the array of bits and then pack?
        # Or we can use a recursive doubling trick?
        
        # Actually, we don't need to precompute the input states as integers if we simulate the network differently.
        # But bit-parallel simulation requires the states to be integers (or big integers).
        
        # Let's try a different approach for initialization:
        # We can generate the bit vectors using a formula.
        # The bit vector for wire i is a periodic sequence.
        # We can generate the integer by using a formula involving geometric series?
        
        # Or, we can just use numpy to create a uint64 array of the bits, and then convert to a Python integer?
        # 2^18 bits is 4096 uint64s.
        # Creating 18 such arrays and converting to integers might be feasible.
        
        # Let's try to generate the initial states using numpy.
        # inputs = np.arange(1 << N, dtype=np.uint64)
        # for i in range(N):
        #     bits = (inputs >> i) & 1
        #     # Pack bits into a big integer?
        #     # Python doesn't have a direct "pack bits to int" that is fast for 262k bits.
        
        # Alternative:
        # We can simulate the network on a *sample* of inputs? No, must be all.
        
        # Let's go back to the block repetition.
        # We can speed up the repetition by doubling the block size.
        # val = block_val
        # shift = block_len
        # while shift < total_bits:
        #     val |= (val << shift)
        #     shift *= 2
        #     # Wait, this works if the pattern is self-similar?
        #     # The pattern is 0^2^i 1^2^i.
        #     # Doubling it gives 0^2^i 1^2^i 0^2^i 1^2^i.
        #     # Yes, this works!
        
        val = 0
        # Construct one block of length 2^(i+1)
        # The ones are at positions i to 2*i - 1 within the block.
        # So block_val = sum(1 << p for p in range(i, 2*i))
        # This is (2^(2*i) - 2^i) // (2^i - 1)? No, it's just a mask.
        # Mask for bits i to 2*i-1 is ((1 << (2*i)) - (1 << i))
        # Wait, if i=1, bits 1,2. Mask = 0b110 = 6.
        # (1 << 2) - (1 << 1) = 4 - 2 = 2. No.
        # Mask = ((1 << (2*i)) - 1) ^ ((1 << i) - 1)
        # For i=1: (1<<2)-1 = 3 (0b11). (1<<1)-1 = 1 (0b01). 3^1 = 2 (0b10). No, we want 0b110.
        # We want bits i through 2i-1.
        # Count is i bits.
        # Mask = ((1 << i) - 1) << i
        # For i=1: (1<<1)-1 = 1. 1 << 1 = 2 (0b10). No, we want bit 1 and bit 2?
        # Wait, range(i, 2*i) for i=1 is range(1, 2) -> [1]. So only bit 1.
        # My previous analysis:
        # i=1: k=0(00)->0, k=1(01)->0, k=2(10)->1, k=3(11)->1.
        # So bits 2 and 3 of the integer are 1.
        # So the block is 0011.
        # The ones are at positions 2 and 3 of the block (0-indexed 0..3).
        # So the mask for the block is 0b1100 = 12.
        # How to get 12 from i=1?
        # Block length 4.
        # Ones at 2,3.
        # Mask = ((1 << i) - 1) << (2*i - i)? No.
        # The ones start at position i? No, for i=1, they start at 2.
        # For i=0, they start at 1.
        # For i=2, k=0..3 -> 0, k=4..7 -> 1. Bits 4..7 are 1.
        # So ones start at 2^i?
        # i=0: start 1 (2^0).
        # i=1: start 2 (2^1).
        # i=2: start 4 (2^2).
        # Yes, ones start at 2^i.
        # And there are 2^i ones.
        # So the mask is ((1 << (2^i)) - 1) << (2^i).
        
        block_val = ((1 << (1 << i)) - 1) << (1 << i)
        
        # Now repeat this block to fill 2^N bits.
        # Total bits = 1 << N.
        # Block len = 1 << (i+1).
        # Repeats = 1 << (N - i - 1).
        
        # Doubling method:
        val = block_val
        shift = 1 << (i + 1)
        total_bits = 1 << N
        while shift < total_bits:
            val |= (val << shift)
            shift *= 2
            
        states[i] = val
        
    # Now simulate the network
    for i, j in comps:
        # Compare wire i and j.
        # min goes to i, max to j.
        # For bit vectors, min is AND, max is OR.
        # new_i = states[i] & states[j]
        # new_j = states[i] | states[j]
        si = states[i]
        sj = states[j]
        states[i] = si & sj
        states[j] = si | sj
        
    # Check if sorted
    # For all k, state[0][k] <= state[1][k] <= ...
    # This means state[j] & ~state[j+1] must be 0 for all j.
    for j in range(N - 1):
        if (states[j] & ~states[j + 1]) != 0:
            return False
            
    return True

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    
    # Seed: Odd-Even Transposition Network
    # This is N * (N-1) / 2 = 18 * 17 / 2 = 153 comparators.
    # Wait, OET is N phases. Each phase has N/2 comparators. Total N^2/2 = 162.
    # Bubble sort is 153.
    # Let's use Bubble Sort as the seed as it is smaller.
    comps = []
    for i in range(N - 1):
        for j in range(N - 1 - i):
            comps.append((j, j + 1))
    
    best_comps = comps[:]
    best_len = len(comps)
    
    # Local Search with Insert-Delete moves
    # Moves:
    # 1. Remove a comparator.
    # 2. Insert a comparator (random position, random wires).
    # 3. Swap adjacent comparators.
    
    # To avoid getting stuck, we allow temporary increases in size (Insert)
    # if we are stuck in a local minimum where no removal is possible.
    
    stuck_count = 0
    
    while time.time() < DEADLINE:
        move_type = random.random()
        
        if move_type < 0.5:
            # Try removal
            if len(comps) > 0:
                k = random.randint(0, len(comps) - 1)
                candidate = comps[:k] + comps[k+1:]
                
                if _verify_network_bit_parallel(candidate):
                    comps = candidate
                    stuck_count = 0
                    if len(comps) < best_len:
                        best_len = len(comps)
                        best_comps = comps[:]
                else:
                    stuck_count += 1
        elif move_type < 0.8:
            # Try insertion
            # Insert a random comparator at a random position
            i = random.randint(0, N - 2)
            j = random.randint(i + 1, N - 1)
            pos = random.randint(0, len(comps))
            
            candidate = comps[:]
            candidate.insert(pos, (i, j))
            
            # We only accept insertion if it leads to a valid network 
            # AND if we are stuck (or just to explore, but we must be careful not to grow indefinitely)
            # Actually, standard hill climbing for minimization should not accept increases 
            # unless it's a simulated annealing or specific escape mechanism.
            # The "Insert-Delete" strategy usually means: 
            # 1. Try to remove. If fail, try to insert a *helpful* comparator?
            # Or: Insert a comparator, then try to remove a *different* one?
            
            # Let's use a simple escape: if stuck_count > 50, try an insertion.
            # But we must ensure we don't grow too large.
            # Let's try: Insert a comparator, then immediately try to remove one (possibly the same or another).
            # This is a "2-move" sequence.
            
            if stuck_count > 20:
                # Try a 2-move sequence: Insert then Remove
                # 1. Insert (i, j) at pos
                candidate = comps[:]
                candidate.insert(pos, (i, j))
                
                # 2. Try to remove a comparator from candidate
                # We can try removing the one we just inserted (backtrack) or a random other one.
                # Let's try removing a random one from candidate.
                if len(candidate) > 0:
                    rem_idx = random.randint(0, len(candidate) - 1)
                    candidate2 = candidate[:rem_idx] + candidate[rem_idx+1:]
                    
                    if _verify_network_bit_parallel(candidate2):
                        # If valid, and if candidate2 is smaller than comps, accept.
                        # If candidate2 is same size, accept to escape local minimum?
                        # If candidate2 is larger, reject.
                        if len(candidate2) <= len(comps):
                            comps = candidate2
                            stuck_count = 0
                            if len(comps) < best_len:
                                best_len = len(comps)
                                best_comps = comps[:]
                        else:
                            # If it grew, we might still accept it temporarily to allow further reduction?
                            # No, let's stick to non-increasing moves for simplicity, 
                            # but allow same-size moves to change the structure.
                            pass
                stuck_count = 0 # Reset stuck count after trying an escape
            else:
                # Just try insertion? No, that increases size.
                # Let's try a swap instead if not stuck.
                if len(comps) >= 2:
                    k = random.randint(0, len(comps) - 2)
                    candidate = comps[:]
                    candidate[k], candidate[k+1] = candidate[k+1], candidate[k]
                    if _verify_network_bit_parallel(candidate):
                        comps = candidate
                        stuck_count = 0
                    else:
                        stuck_count += 1
        else:
            # Try swap (if not handled above)
            if len(comps) >= 2:
                k = random.randint(0, len(comps) - 2)
                candidate = comps[:]
                candidate[k], candidate[k+1] = candidate[k+1], candidate[k]
                if _verify_network_bit_parallel(candidate):
                    comps = candidate
                    stuck_count = 0
                else:
                    stuck_count += 1

    return best_comps
# EVOLVE-BLOCK-END
