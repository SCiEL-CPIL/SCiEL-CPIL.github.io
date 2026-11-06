For each removed triple t (in random order):
  If adding t back doesn't violate: add it
  Else: for each constraint c that t would violate:
    Find a triple t2 in c that is currently present
    If removing t2 and adding t doesn't create new violations: do the swap
