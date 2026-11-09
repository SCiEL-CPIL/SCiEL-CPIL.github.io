"""Cap set greedy construction with a refined priority heuristic."""
import hashlib

def priority(vector, n):
    # Calculate digit counts
    c0 = vector.count(0)
    c1 = vector.count(1)
    c2 = vector.count(2)
    
    # Balance penalty: vectors with digits distributed evenly are prioritized
    target = n / 3.0
    penalty = (c0 - target) ** 2 + (c1 - target) ** 2 + (c2 - target) ** 2
    
    # Deterministic pseudo-random noise based on the vector content
    # This breaks ties in a way that avoids systematic biases of lex or sum ordering
    # within the same balance class.
    v_str = str(vector)
    h = hashlib.md5(v_str.encode('utf-8')).hexdigest()
    noise = int(h, 16) % 1000000 / 1000000.0
    
    # Priority:
    # 1. Primary: Minimize balance penalty (higher score for lower penalty)
    # 2. Secondary: Randomized order to explore different greedy paths
    # We scale penalty heavily to ensure balance is the dominant factor.
    return -penalty * 1000000.0 + noise
