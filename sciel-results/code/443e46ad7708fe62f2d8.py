    to_delete = set(candidates)
    new_word = []
    for i, sym in enumerate(word):
        if i in to_delete:
             # Need to check if it's still valid?
             # The validity might have changed if a neighbor was deleted.
             # But if we delete multiple, we might break validity.
             # Safe approach: Delete one, re-compute can_delete, repeat?
             # That's slow.
             pass
    