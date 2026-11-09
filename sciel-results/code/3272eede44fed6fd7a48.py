import time
import math
import random
from itertools import product

_START_TIME = time.time()
TIME_LIMIT = 280

# Configuration for Dynamic Bucket Repulsion
_NUM_HASHES = 20
_BUCKET_SIZE = 128
_lambda_repulsion = 0.5

# Pre-generate hash weights
_random_state = random.Random(42)
_HASH_WEIGHTS = []
for h in range(_NUM_HASHES):
    # Weights for up to n=10 (3^10 ~ 59000 vectors)
    # If n is larger, we just take the first n weights
    weights = [_random_state.randint(0, 2) for _ in range(20)]
    _HASH_WEIGHTS.append(weights)

# Global state for bucket counts
# bucket_counts[h][b]
_bucket_counts = [[0] * _BUCKET_SIZE for _ in range(_NUM_HASHES)]

def _compute_hash(v, n, h_id):
    """Compute the hash bucket index for vector v using hash function h_id."""
    weights = _HASH_WEIGHTS[h_id]
    h = 0
    for i in range(n):
        h = (h + v[i] * weights[i]) % _BUCKET_SIZE
    return h

def priority(vector, n):
    """
    Dynamic Bucket Repulsion Priority.
    
    This function penalizes vectors that fall into hash buckets already occupied
    by previously processed vectors. This enforces diversity and approximates a
    greedy set cover, which helps in finding large cap sets.
    """
    # 1. Compute structural base score (simple entropy/balance)
    c0 = 0
    c1 = 0
    c2 = 0
    for v in vector:
        if v == 0: c0 += 1
        elif v == 1: c1 += 1
        else: c2 += 1
    
    # Balance score: prefer balanced vectors
    target = n / 3.0
    balance = (c0 - target)**2 + (c1 - target)**2 + (c2 - target)**2
    base_score = -balance / (n + 1)
    
    # 2. Compute dynamic repulsion score
    density = 0
    for h_id in range(_NUM_HASHES):
        bucket = _compute_hash(vector, n, h_id)
        density += _bucket_counts[h_id][bucket]
        
        # Update the count for future calls
        _bucket_counts[h_id][bucket] += 1
    
    # 3. Combine scores
    # Higher base_score is better.
    # Higher density is worse (repulsion).
    score = base_score - _lambda_repulsion * density
    
    return score
