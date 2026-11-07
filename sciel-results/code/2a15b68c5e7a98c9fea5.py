import struct
import time

# State layout (1024 bytes total):
#   0..511: 512 x uint16 counters for items 0..511 (but we only have 0..399, so 400 used)
#   512..1023: 512 x uint16 = 256 x uint32? No, let's use a different approach.
#
# Better layout:
#   0..399: 400 x uint16 exact counters for items 0..399 (800 bytes)
#   400..1023: 624 bytes remaining for a small CM sketch for overflow/verification?
#
# Wait, 400 * 2 = 800 bytes. We have 1024 - 800 = 224 bytes left.
# 224 bytes = 112 uint16 or 56 uint32.
#
# Actually, let's think about this differently.
# The max count per item in 1024 items is 1024, which fits in uint16 (max 65535).
# So 400 uint16 counters use 800 bytes.
# Remaining 224 bytes can be used for a small CM sketch or just padding.
#
# But the key insight: if we store EXACT counts for all 400 items, the error is 0!
# The only issue is whether we can fit them and whether update/query are fast enough.
#
# 400 items * 2 bytes = 800 bytes. Fits in 1024.
# Update: increment the counter for the given item. O(1).
# Query: return the counter for the given item. O(1).
#
# This gives EXACT frequency estimates, so the score would be 0.0 (perfect).
# The only risk is performance, but simple uint16 increments are very fast.
#
# Let's verify: items are 0..399, so 400 distinct items.
# Each stream has 1024 items, so max count for any item is 1024 < 65535.
# Three streams total: but each stream is processed separately with a fresh state?
# No, re-reading: "Each 256-item chunk runs in a fresh interpreter"
# So the state persists across chunks within a stream, but each stream starts fresh?
# Actually: "Three insertion-only streams of 1024 items"
# "Each 256-item chunk runs in a fresh interpreter (15 s limit), so only the returned bytes persist"
# So within one stream, 4 chunks of 256 items, state carries over between chunks.
# Across streams, does state carry over? "All 400 items are queried once, on the final state."
# This suggests all three streams are processed sequentially into the same state? Or separate states?
#
# Re-reading more carefully: "Three insertion-only streams of 1024 items" - these are three separate streams.
# "All 400 items are queried once, on the final state" - singular "the final state".
# This likely means all three streams are fed into the same state sequentially.
# So total items = 3 * 1024 = 3072.
# Max count per item could be up to 3072 if all items were the same, but with uniform/Zipf, max is much less.
# 3072 < 65535, so uint16 still works.
#
# Wait, but the scoring says "per stream, mean over the 400 items of |estimate - true count| / 1024"
# So each stream has its own true counts, and we query after each stream? Or after all?
# "All 400 items are queried once, on the final state" - this is ambiguous.
#
# Let me re-read: "Three insertion-only streams of 1024 items: ... Each 256-item chunk runs in a fresh interpreter"
# "All 400 items are queried once, on the final state."
#
# I think the evaluation is:
# 1. Start with initial state (1024 zero bytes)
# 2. Process stream 1 (1024 items in 4 chunks of 256)
# 3. Query all 400 items -> get estimates for stream 1
# 4. Process stream 2 (1024 items in 4 chunks of 256) - state continues from after stream 1? Or reset?
# 5. Query all 400 items -> get estimates for stream 2
# 6. Process stream 3
# 7. Query all 400 items -> get estimates for stream 3
#
# If state accumulates across streams, then after stream 1, counts reflect stream 1.
# After stream 2, counts reflect stream 1 + stream 2. But we want just stream 2's counts.
# That would make exact counters not work directly.
#
# Hmm, but "on the final state" suggests only one query at the end.
# Let me assume the state is reset between streams, or that we only query at the very end.
#
# Actually, the safest interpretation given "per stream" scoring: each stream is evaluated independently.
# So state is likely reset between streams, OR the evaluator handles this.
#
# Given the ambiguity, let's go with exact counters assuming state is fresh per stream.
# If state accumulates, we'd need a different approach, but let's try exact first.

def update(state, item):
    # item is 0..399
    # state is 1024 bytes
    # Use first 800 bytes as 400 uint16 counters
    counters = struct.unpack_from('400H', state)
    counters = list(counters)
    counters[item] = min(counters[item] + 1, 65535)
    result = bytearray(state)
    struct.pack_into('400H', result, 0, *counters)
    return bytes(result)

def query(state, item):
    # item is 0..399
    val = struct.unpack_from('H', state, item * 2)[0]
    return val
