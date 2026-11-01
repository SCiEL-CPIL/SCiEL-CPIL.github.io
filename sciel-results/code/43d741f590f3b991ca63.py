
Wait, I need to double check: the seed's rank-55 — does it actually verify? Let me reconsider. The seed returns rank 55 only if it passes `_verify`; otherwise schoolbook (60). The seed's rank-55 Strassen construction is plausibly valid (score 1.0 = 55, since seed is baseline 1.0 and value 55 → score 1.0 means value=55, seed value=55). So island best value = 55. To beat it I need R < 55, i.e. 50-54. My search tries 50-59, so if it finds 50-54 it beats 55.

Let me finalize.
