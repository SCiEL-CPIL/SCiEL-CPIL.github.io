"""Cap set greedy construction with Hamming-weight-balanced priority and time guard."""
import time
import zlib

_DEADLINE = 330.0 - 15.0
_START = time.time()

def priority(vector, n):
    # Assign higher priority to vectors with Hamming weight close to n/2.
    # Use CRC32 hash as tiebreaker to avoid lexicographic bias.
    weight = sum(1 for x in vector if x != 0)
    target = n / 2.0
    weight_score = -(weight - target) ** 2
    h = zlib.crc32(vector) & 0xFFFFFFFF
    hash_score = (h / 0xFFFFFFFF) * 0.01
    return weight_score + hash_score
